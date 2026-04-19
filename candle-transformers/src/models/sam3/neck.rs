use candle::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder};

use super::config::NeckConfig;
use super::vitdet::ViTDetTrunkOutput;

#[derive(Debug, Clone)]
pub struct VisualBackboneOutput {
    pub backbone_fpn: Vec<Tensor>,
    pub vision_pos_enc: Vec<Tensor>,
    pub sam2_backbone_fpn: Option<Vec<Tensor>>,
    pub sam2_pos_enc: Option<Vec<Tensor>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PyramidStageKind {
    UpsampleX4,
    UpsampleX2,
    Identity,
    DownsampleX2,
}

impl PyramidStageKind {
    fn from_scale_factor(scale_factor: f32) -> Result<Self> {
        if approx_eq(scale_factor, 4.0) {
            Ok(Self::UpsampleX4)
        } else if approx_eq(scale_factor, 2.0) {
            Ok(Self::UpsampleX2)
        } else if approx_eq(scale_factor, 1.0) {
            Ok(Self::Identity)
        } else if approx_eq(scale_factor, 0.5) {
            Ok(Self::DownsampleX2)
        } else {
            candle::bail!("unsupported sam3 neck scale factor {scale_factor}")
        }
    }
}

fn approx_eq(lhs: f32, rhs: f32) -> bool {
    (lhs - rhs).abs() < 1e-6
}

#[derive(Debug)]
struct FeaturePyramidStage {
    kind: PyramidStageKind,
    upsample0: Option<ConvTranspose2d>,
    upsample1: Option<ConvTranspose2d>,
    conv_1x1: Conv2d,
    conv_3x3: Conv2d,
}

impl FeaturePyramidStage {
    fn new(
        kind: PyramidStageKind,
        input_channels: usize,
        d_model: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let upsample_cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv_1x1_cfg = Conv2dConfig::default();
        let conv_3x3_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let (upsample0, upsample1, conv_1x1_in_channels) = match kind {
            PyramidStageKind::UpsampleX4 => {
                let mid_channels = input_channels / 2;
                if mid_channels == 0 {
                    candle::bail!(
                        "sam3 neck upsample-x4 stage requires positive mid channels, got input {input_channels}"
                    )
                }
                (
                    Some(candle_nn::conv_transpose2d(
                        input_channels,
                        mid_channels,
                        2,
                        upsample_cfg,
                        vb.pp("dconv_2x2_0"),
                    )?),
                    Some(candle_nn::conv_transpose2d(
                        mid_channels,
                        d_model,
                        2,
                        upsample_cfg,
                        vb.pp("dconv_2x2_1"),
                    )?),
                    d_model,
                )
            }
            PyramidStageKind::UpsampleX2 => {
                let mid_channels = input_channels / 2;
                if mid_channels == 0 {
                    candle::bail!(
                        "sam3 neck upsample-x2 stage requires positive mid channels, got input {input_channels}"
                    )
                }
                (
                    Some(candle_nn::conv_transpose2d(
                        input_channels,
                        mid_channels,
                        2,
                        upsample_cfg,
                        vb.pp("dconv_2x2"),
                    )?),
                    None,
                    mid_channels,
                )
            }
            PyramidStageKind::Identity | PyramidStageKind::DownsampleX2 => {
                (None, None, input_channels)
            }
        };
        let conv_1x1 = candle_nn::conv2d(
            conv_1x1_in_channels,
            d_model,
            1,
            conv_1x1_cfg,
            vb.pp("conv_1x1"),
        )?;
        let conv_3x3 = candle_nn::conv2d(d_model, d_model, 3, conv_3x3_cfg, vb.pp("conv_3x3"))?;
        Ok(Self {
            kind,
            upsample0,
            upsample1,
            conv_1x1,
            conv_3x3,
        })
    }

    fn forward(&self, feature_map: &Tensor) -> Result<Tensor> {
        let feature_map = match self.kind {
            PyramidStageKind::UpsampleX4 => self
                .upsample1
                .as_ref()
                .expect("upsample-x4 second stage must exist")
                .forward(
                    &self
                        .upsample0
                        .as_ref()
                        .expect("upsample-x4 first stage must exist")
                        .forward(feature_map)?
                        .gelu_erf()?,
                )?,
            PyramidStageKind::UpsampleX2 => self
                .upsample0
                .as_ref()
                .expect("upsample-x2 stage must exist")
                .forward(feature_map)?,
            PyramidStageKind::Identity => feature_map.clone(),
            PyramidStageKind::DownsampleX2 => feature_map.max_pool2d_with_stride(2, 2)?,
        };
        let feature_map = self.conv_1x1.forward(&feature_map)?;
        self.conv_3x3.forward(&feature_map)
    }
}

#[derive(Debug)]
pub struct Sam3DualViTDetNeck {
    config: NeckConfig,
    stages: Vec<FeaturePyramidStage>,
    sam2_stages: Option<Vec<FeaturePyramidStage>>,
}

impl Sam3DualViTDetNeck {
    pub fn new(config: &NeckConfig, vb: VarBuilder) -> Result<Self> {
        let stages = build_stages(config, vb.pp("convs"))?;
        let sam2_stages = if config.add_sam2_neck {
            Some(build_stages(config, vb.pp("sam2_convs"))?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            stages,
            sam2_stages,
        })
    }

    pub fn config(&self) -> &NeckConfig {
        &self.config
    }

    pub fn forward(&self, trunk: &ViTDetTrunkOutput) -> Result<VisualBackboneOutput> {
        let Some(feature_map) = trunk.stage_features.last() else {
            candle::bail!("sam3 neck expects at least one trunk feature map")
        };
        let feature_map = feature_map.permute((0, 3, 1, 2))?;
        let backbone_fpn = self.forward_branch(&self.stages, &feature_map)?;
        let vision_pos_enc = build_position_encodings(&backbone_fpn, self.config.d_model)?;
        let (sam2_backbone_fpn, sam2_pos_enc) = match &self.sam2_stages {
            Some(stages) => {
                let branch = self.forward_branch(stages, &feature_map)?;
                let pos = vision_pos_enc.iter().map(Tensor::clone).collect();
                (Some(branch), Some(pos))
            }
            None => (None, None),
        };
        Ok(VisualBackboneOutput {
            backbone_fpn,
            vision_pos_enc,
            sam2_backbone_fpn,
            sam2_pos_enc,
        })
    }

    fn forward_branch(
        &self,
        stages: &[FeaturePyramidStage],
        feature_map: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let mut levels = Vec::with_capacity(stages.len());
        for stage in stages {
            levels.push(stage.forward(feature_map)?);
        }
        if self.config.scalp > levels.len() {
            candle::bail!(
                "sam3 neck scalp {} exceeds number of generated levels {}",
                self.config.scalp,
                levels.len()
            )
        }
        levels.truncate(levels.len() - self.config.scalp);
        Ok(levels)
    }
}

fn build_stages(config: &NeckConfig, vb: VarBuilder) -> Result<Vec<FeaturePyramidStage>> {
    let mut stages = Vec::with_capacity(config.scale_factors.len());
    for (index, scale_factor) in config.scale_factors.iter().copied().enumerate() {
        let kind = PyramidStageKind::from_scale_factor(scale_factor)?;
        let input_channels = stage_input_channels(kind, config.d_model)?;
        stages.push(FeaturePyramidStage::new(
            kind,
            input_channels,
            config.d_model,
            vb.pp(index),
        )?);
    }
    Ok(stages)
}

fn stage_input_channels(kind: PyramidStageKind, d_model: usize) -> Result<usize> {
    match kind {
        PyramidStageKind::UpsampleX4
        | PyramidStageKind::UpsampleX2
        | PyramidStageKind::Identity
        | PyramidStageKind::DownsampleX2 => {
            let trunk_channels = d_model * 4;
            if trunk_channels == 0 {
                candle::bail!("sam3 neck d_model must be positive")
            }
            Ok(trunk_channels)
        }
    }
}

fn build_position_encodings(features: &[Tensor], d_model: usize) -> Result<Vec<Tensor>> {
    let mut encodings = Vec::with_capacity(features.len());
    for feature in features {
        encodings.push(build_2d_sine_position_encoding(feature, d_model)?);
    }
    Ok(encodings)
}

fn build_2d_sine_position_encoding(feature: &Tensor, d_model: usize) -> Result<Tensor> {
    let (batch_size, channels, height, width) = feature.dims4()?;
    if channels != d_model {
        candle::bail!("sam3 neck expected projected feature width {d_model}, got {channels}")
    }
    if d_model % 2 != 0 {
        candle::bail!("sam3 neck position encoding requires even d_model, got {d_model}")
    }
    let num_pos_feats = d_model / 2;
    if num_pos_feats % 2 != 0 {
        candle::bail!("sam3 neck position encoding requires d_model divisible by 4, got {d_model}")
    }
    let device = feature.device();
    let dtype = feature.dtype();
    let temperature = 10_000f32;
    let scale = 2.0 * std::f32::consts::PI;
    let eps = 1e-6f32;
    let mut dim_t = Vec::with_capacity(num_pos_feats);
    for idx in 0..num_pos_feats {
        let exponent = 2.0 * (idx / 2) as f32 / num_pos_feats as f32;
        dim_t.push(temperature.powf(exponent));
    }
    let mut encoding = vec![0f32; d_model * height * width];
    for y in 0..height {
        let y_pos = ((y + 1) as f32 / (height as f32 + eps)) * scale;
        for x in 0..width {
            let x_pos = ((x + 1) as f32 / (width as f32 + eps)) * scale;
            for idx in 0..num_pos_feats {
                let div = dim_t[idx];
                let y_value = if idx % 2 == 0 {
                    (y_pos / div).sin()
                } else {
                    (y_pos / div).cos()
                };
                let x_value = if idx % 2 == 0 {
                    (x_pos / div).sin()
                } else {
                    (x_pos / div).cos()
                };
                let spatial_index = y * width + x;
                encoding[idx * height * width + spatial_index] = y_value;
                encoding[(num_pos_feats + idx) * height * width + spatial_index] = x_value;
            }
        }
    }
    let encoding = Tensor::from_slice(&encoding, (1, d_model, height, width), device)?;
    encoding.repeat((batch_size, 1, 1, 1))?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use candle::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;

    use super::{Sam3DualViTDetNeck, VisualBackboneOutput};
    use crate::models::sam3::{NeckConfig, ViTDetTrunkOutput};

    fn test_device() -> Result<Device> {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0)
        }
        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            Device::new_metal(0)
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Ok(Device::Cpu)
        }
    }

    #[test]
    fn neck_projects_single_trunk_feature_map_to_three_levels() -> Result<()> {
        let device = test_device()?;
        let config = NeckConfig {
            d_model: 4,
            scale_factors: [4.0, 2.0, 1.0, 0.5],
            scalp: 1,
            add_sam2_neck: false,
        };
        let vb = VarBuilder::from_tensors(
            neck_weights(&device, config.d_model, false)?,
            DType::F32,
            &device,
        );
        let neck = Sam3DualViTDetNeck::new(&config, vb)?;
        let trunk = ViTDetTrunkOutput {
            stage_features: vec![Tensor::zeros((1, 2, 2, 16), DType::F32, &device)?],
        };
        let out = neck.forward(&trunk)?;
        assert_visual_shapes(&out, &[(1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2)])?;
        assert!(out.sam2_backbone_fpn.is_none());
        assert!(out.sam2_pos_enc.is_none());
        Ok(())
    }

    #[test]
    fn neck_can_load_optional_sam2_branch() -> Result<()> {
        let device = test_device()?;
        let config = NeckConfig {
            d_model: 4,
            scale_factors: [4.0, 2.0, 1.0, 0.5],
            scalp: 1,
            add_sam2_neck: true,
        };
        let vb = VarBuilder::from_tensors(
            neck_weights(&device, config.d_model, true)?,
            DType::F32,
            &device,
        );
        let neck = Sam3DualViTDetNeck::new(&config, vb)?;
        let trunk = ViTDetTrunkOutput {
            stage_features: vec![Tensor::zeros((1, 2, 2, 16), DType::F32, &device)?],
        };
        let out = neck.forward(&trunk)?;
        assert_eq!(out.backbone_fpn.len(), 3);
        assert_eq!(out.sam2_backbone_fpn.as_ref().map(Vec::len), Some(3));
        assert_eq!(out.sam2_pos_enc.as_ref().map(Vec::len), Some(3));
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_neck_last_level_matches_upstream() -> Result<()> {
        let device = test_device()?;
        let output = run_interactive_visual_fixture_neck(&device)?;
        let expected = load_interactive_visual_fixture_tensors("fixture.safetensors", &device)?;
        assert_tensor_close(
            output
                .backbone_fpn
                .last()
                .expect("neck should emit at least one FPN level"),
            fixture_tensor(&expected, "vision.backbone_fpn.last")?,
            1e-5,
            "vision.backbone_fpn.last",
        )
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_position_encoding_matches_upstream() -> Result<()> {
        let device = test_device()?;
        let output = run_interactive_visual_fixture_neck(&device)?;
        let expected = load_interactive_visual_fixture_tensors("fixture.safetensors", &device)?;
        assert_tensor_close(
            output
                .vision_pos_enc
                .last()
                .expect("neck should emit at least one position encoding"),
            fixture_tensor(&expected, "vision.vision_pos_enc.last")?,
            1e-5,
            "vision.vision_pos_enc.last",
        )
    }

    fn assert_visual_shapes(
        out: &VisualBackboneOutput,
        expected: &[(usize, usize, usize, usize)],
    ) -> Result<()> {
        assert_eq!(out.backbone_fpn.len(), expected.len());
        assert_eq!(out.vision_pos_enc.len(), expected.len());
        for (feature, dims) in out.backbone_fpn.iter().zip(expected.iter()) {
            assert_eq!(feature.dims4()?, *dims);
        }
        for (pos, dims) in out.vision_pos_enc.iter().zip(expected.iter()) {
            assert_eq!(pos.dims4()?, *dims);
        }
        Ok(())
    }

    fn run_interactive_visual_fixture_neck(device: &Device) -> Result<VisualBackboneOutput> {
        let weights =
            load_interactive_visual_fixture_tensors("vision_backbone_weights.safetensors", device)?;
        let fixture = load_interactive_visual_fixture_tensors("fixture.safetensors", device)?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let neck = Sam3DualViTDetNeck::new(&NeckConfig::default(), vb)?;
        let trunk_last = fixture_tensor(&fixture, "vision.trunk.last")?.permute((0, 2, 3, 1))?;
        let trunk = ViTDetTrunkOutput {
            stage_features: vec![trunk_last],
        };
        neck.forward(&trunk)
    }

    fn interactive_visual_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_interactive_visual_seed")
    }

    fn load_interactive_visual_fixture_tensors(
        file_name: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let path = interactive_visual_fixture_dir().join(file_name);
        candle::safetensors::load(&path, device).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load interactive visual fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_tensor<'a>(fixture: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
        fixture.get(key).ok_or_else(|| {
            candle::Error::Msg(format!(
                "interactive visual fixture is missing tensor `{key}`"
            ))
        })
    }

    fn assert_tensor_close(
        actual: &Tensor,
        expected: &Tensor,
        atol: f32,
        name: &str,
    ) -> Result<()> {
        if actual.dims() != expected.dims() {
            candle::bail!(
                "{name}: shape mismatch actual={:?} expected={:?}",
                actual.dims(),
                expected.dims()
            );
        }
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_abs_diff = 0f32;
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            max_abs_diff = max_abs_diff.max((lhs - rhs).abs());
        }
        if max_abs_diff > atol {
            candle::bail!("{name}: max_abs_diff={max_abs_diff:.8} exceeded atol={atol:.8}");
        }
        Ok(())
    }

    fn neck_weights(
        device: &Device,
        d_model: usize,
        include_sam2: bool,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        let trunk_channels = d_model * 4;
        let mid_channels = trunk_channels / 2;
        add_branch_weights(
            &mut tensors,
            "convs",
            device,
            d_model,
            trunk_channels,
            mid_channels,
        )?;
        if include_sam2 {
            add_branch_weights(
                &mut tensors,
                "sam2_convs",
                device,
                d_model,
                trunk_channels,
                mid_channels,
            )?;
        }
        Ok(tensors)
    }

    fn add_branch_weights(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        device: &Device,
        d_model: usize,
        trunk_channels: usize,
        mid_channels: usize,
    ) -> Result<()> {
        tensors.insert(
            format!("{prefix}.0.dconv_2x2_0.weight"),
            Tensor::zeros((trunk_channels, mid_channels, 2, 2), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.dconv_2x2_1.weight"),
            Tensor::zeros((mid_channels, d_model, 2, 2), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.dconv_2x2_0.bias"),
            Tensor::zeros(mid_channels, DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.dconv_2x2_1.bias"),
            Tensor::zeros(d_model, DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.conv_1x1.weight"),
            Tensor::zeros((d_model, d_model, 1, 1), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.conv_1x1.bias"),
            Tensor::zeros(d_model, DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.conv_3x3.weight"),
            Tensor::zeros((d_model, d_model, 3, 3), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.0.conv_3x3.bias"),
            Tensor::zeros(d_model, DType::F32, device)?,
        );

        tensors.insert(
            format!("{prefix}.1.dconv_2x2.weight"),
            Tensor::zeros((trunk_channels, mid_channels, 2, 2), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.1.dconv_2x2.bias"),
            Tensor::zeros(mid_channels, DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.1.conv_1x1.weight"),
            Tensor::zeros((d_model, mid_channels, 1, 1), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.1.conv_1x1.bias"),
            Tensor::zeros(d_model, DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.1.conv_3x3.weight"),
            Tensor::zeros((d_model, d_model, 3, 3), DType::F32, device)?,
        );
        tensors.insert(
            format!("{prefix}.1.conv_3x3.bias"),
            Tensor::zeros(d_model, DType::F32, device)?,
        );

        for idx in [2usize, 3usize] {
            tensors.insert(
                format!("{prefix}.{idx}.conv_1x1.weight"),
                Tensor::zeros((d_model, trunk_channels, 1, 1), DType::F32, device)?,
            );
            tensors.insert(
                format!("{prefix}.{idx}.conv_1x1.bias"),
                Tensor::zeros(d_model, DType::F32, device)?,
            );
            tensors.insert(
                format!("{prefix}.{idx}.conv_3x3.weight"),
                Tensor::zeros((d_model, d_model, 3, 3), DType::F32, device)?,
            );
            tensors.insert(
                format!("{prefix}.{idx}.conv_3x3.bias"),
                Tensor::zeros(d_model, DType::F32, device)?,
            );
        }
        Ok(())
    }
}
