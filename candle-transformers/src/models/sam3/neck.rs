use std::collections::HashMap;
use std::sync::Mutex;

use candle::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder};

use super::config::NeckConfig;
use super::profiling;
use super::torch_ops::position::build_2d_sine_position_encoding_grid;
use super::tracker::add_tensor_memory;
use super::vitdet::ViTDetTrunkOutput;

#[derive(Debug, Clone)]
pub struct TrackerVisualSequences {
    pub feat_sizes: Vec<(usize, usize)>,
    pub vision_feats: Vec<Tensor>,
    pub vision_pos_embeds: Vec<Tensor>,
}

#[derive(Debug, Clone)]
pub struct VisualBackboneOutput {
    pub backbone_fpn: Vec<Tensor>,
    pub vision_pos_enc: Vec<Tensor>,
    pub sam2_backbone_fpn: Option<Vec<Tensor>>,
    pub sam2_pos_enc: Option<Vec<Tensor>>,
    pub tracker_sequences: Option<TrackerVisualSequences>,
    pub tracker_sam2_sequences: Option<TrackerVisualSequences>,
}

impl VisualBackboneOutput {
    pub(crate) fn ensure_tracker_sequences(&mut self) -> Result<()> {
        if self.tracker_sequences.is_none() {
            self.tracker_sequences = Some(build_tracker_visual_sequences(
                self.backbone_fpn.as_slice(),
                self.vision_pos_enc.as_slice(),
            )?);
        }
        if self.tracker_sam2_sequences.is_none() {
            self.tracker_sam2_sequences = match (&self.sam2_backbone_fpn, &self.sam2_pos_enc) {
                (Some(backbone_fpn), Some(vision_pos_enc)) => Some(build_tracker_visual_sequences(
                    backbone_fpn.as_slice(),
                    vision_pos_enc.as_slice(),
                )?),
                _ => None,
            };
        }
        Ok(())
    }

    pub(crate) fn strip_tracker_sequences(&mut self) {
        self.tracker_sequences = None;
        self.tracker_sam2_sequences = None;
    }

    pub(crate) fn memory_bytes(&self) -> (usize, usize) {
        let mut cpu = 0usize;
        let mut device = 0usize;
        for tensor in self.backbone_fpn.iter() {
            add_tensor_memory(tensor, &mut cpu, &mut device);
        }
        for tensor in self.vision_pos_enc.iter() {
            add_tensor_memory(tensor, &mut cpu, &mut device);
        }
        if let Some(levels) = self.sam2_backbone_fpn.as_ref() {
            for tensor in levels.iter() {
                add_tensor_memory(tensor, &mut cpu, &mut device);
            }
        }
        if let Some(levels) = self.sam2_pos_enc.as_ref() {
            for tensor in levels.iter() {
                add_tensor_memory(tensor, &mut cpu, &mut device);
            }
        }
        if let Some(sequences) = self.tracker_sequences.as_ref() {
            for tensor in sequences.vision_feats.iter() {
                add_tensor_memory(tensor, &mut cpu, &mut device);
            }
            for tensor in sequences.vision_pos_embeds.iter() {
                add_tensor_memory(tensor, &mut cpu, &mut device);
            }
        }
        if let Some(sequences) = self.tracker_sam2_sequences.as_ref() {
            for tensor in sequences.vision_feats.iter() {
                add_tensor_memory(tensor, &mut cpu, &mut device);
            }
            for tensor in sequences.vision_pos_embeds.iter() {
                add_tensor_memory(tensor, &mut cpu, &mut device);
            }
        }
        (cpu, device)
    }
}

pub(crate) fn build_tracker_visual_sequences(
    backbone_fpn: &[Tensor],
    vision_pos_enc: &[Tensor],
) -> Result<TrackerVisualSequences> {
    let _range = profiling::range("sam3.neck.tracker_sequences.primary");
    let feat_sizes = backbone_fpn
        .iter()
        .zip(vision_pos_enc.iter())
        .map(|(feat, pos)| {
            let (_, feat_channels, feat_h, feat_w) = feat.dims4()?;
            let pos_shape = pos.dims4()?;
            if pos_shape != (1, feat_channels, feat_h, feat_w) {
                candle::bail!(
                    "tracker expected matching feature/pos shapes, got ({feat_channels}, {feat_h}, {feat_w}) and {pos_shape:?}"
                )
            }
            Ok((feat_h, feat_w))
        })
        .collect::<Result<Vec<_>>>()?;
    let vision_feats = backbone_fpn
        .iter()
        .map(|feat| {
            feat.permute((2, 3, 0, 1))?.reshape((
                feat.dim(2)? * feat.dim(3)?,
                feat.dim(0)?,
                feat.dim(1)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let vision_pos_embeds = vision_pos_enc
        .iter()
        .map(|pos| {
            pos.permute((2, 3, 0, 1))?.reshape((
                pos.dim(2)? * pos.dim(3)?,
                pos.dim(0)?,
                pos.dim(1)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TrackerVisualSequences {
        feat_sizes,
        vision_feats,
        vision_pos_embeds,
    })
}

fn build_tracker_visual_sequences_reusing_position_layouts(
    backbone_fpn: &[Tensor],
    primary_sequences: &TrackerVisualSequences,
) -> Result<TrackerVisualSequences> {
    let _range = profiling::range("sam3.neck.tracker_sequences.sam2_reuse_position");
    if backbone_fpn.len() != primary_sequences.feat_sizes.len()
        || backbone_fpn.len() != primary_sequences.vision_pos_embeds.len()
    {
        candle::bail!(
            "tracker expected matching SAM2 feature and primary position levels, got {} features, {} sizes, and {} position layouts",
            backbone_fpn.len(),
            primary_sequences.feat_sizes.len(),
            primary_sequences.vision_pos_embeds.len()
        )
    }
    for (level, (feature, expected_hw)) in backbone_fpn
        .iter()
        .zip(primary_sequences.feat_sizes.iter())
        .enumerate()
    {
        let (_, _, height, width) = feature.dims4()?;
        if (height, width) != *expected_hw {
            candle::bail!(
                "tracker SAM2 feature level {level} has spatial shape {height}x{width}, expected {}x{}",
                expected_hw.0,
                expected_hw.1
            )
        }
    }
    let vision_feats = backbone_fpn
        .iter()
        .map(|feat| {
            feat.permute((2, 3, 0, 1))?.reshape((
                feat.dim(2)? * feat.dim(3)?,
                feat.dim(0)?,
                feat.dim(1)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TrackerVisualSequences {
        feat_sizes: primary_sequences.feat_sizes.clone(),
        vision_feats,
        vision_pos_embeds: primary_sequences.vision_pos_embeds.clone(),
    })
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
    profile_name: String,
}

impl FeaturePyramidStage {
    fn new(
        kind: PyramidStageKind,
        branch: &str,
        stage_index: usize,
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
            profile_name: format!("sam3.neck.{branch}.stage.{stage_index}"),
        })
    }

    fn forward(&self, feature_map: &Tensor) -> Result<Tensor> {
        let _stage_range = profiling::range(&self.profile_name);
        let feature_map = match self.kind {
            PyramidStageKind::UpsampleX4 => {
                let feature_map = {
                    let _range = profiling::range("sam3.neck.upsample0");
                    self.upsample0
                        .as_ref()
                        .expect("upsample-x4 first stage must exist")
                        .forward(feature_map)?
                };
                let feature_map = {
                    let _range = profiling::range("sam3.neck.upsample_gelu");
                    feature_map.gelu_erf()?
                };
                let _range = profiling::range("sam3.neck.upsample1");
                self.upsample1
                    .as_ref()
                    .expect("upsample-x4 second stage must exist")
                    .forward(&feature_map)?
            }
            PyramidStageKind::UpsampleX2 => {
                let _range = profiling::range("sam3.neck.upsample0");
                self.upsample0
                    .as_ref()
                    .expect("upsample-x2 stage must exist")
                    .forward(feature_map)?
            }
            PyramidStageKind::Identity => feature_map.clone(),
            PyramidStageKind::DownsampleX2 => {
                let _range = profiling::range("sam3.neck.downsample_pool");
                feature_map.max_pool2d_with_stride(2, 2)?
            }
        };
        let feature_map = {
            let _range = profiling::range("sam3.neck.conv1x1");
            self.conv_1x1.forward(&feature_map)?
        };
        let _range = profiling::range("sam3.neck.conv3x3");
        self.conv_3x3.forward(&feature_map)
    }

    fn output_shape(&self, height: usize, width: usize) -> (usize, usize) {
        match self.kind {
            PyramidStageKind::UpsampleX4 => (height * 4, width * 4),
            PyramidStageKind::UpsampleX2 => (height * 2, width * 2),
            PyramidStageKind::Identity => (height, width),
            PyramidStageKind::DownsampleX2 => (height / 2, width / 2),
        }
    }
}

#[derive(Debug)]
pub struct Sam3DualViTDetNeck {
    config: NeckConfig,
    stages: Vec<FeaturePyramidStage>,
    sam2_stages: Option<Vec<FeaturePyramidStage>>,
    position_encoding_cache: Mutex<HashMap<PositionEncodingCacheKey, Tensor>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PositionEncodingCacheKey {
    device: String,
    dtype: String,
    d_model: usize,
    height: usize,
    width: usize,
}

impl Sam3DualViTDetNeck {
    pub fn new(config: &NeckConfig, vb: VarBuilder) -> Result<Self> {
        let stages = build_stages(config, "primary", vb.pp("convs"))?;
        let sam2_stages = if config.add_sam2_neck {
            Some(build_stages(config, "sam2", vb.pp("sam2_convs"))?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            stages,
            sam2_stages,
            position_encoding_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn config(&self) -> &NeckConfig {
        &self.config
    }

    pub(crate) fn prime_position_encoding_cache(
        &self,
        device: &Device,
        dtype: DType,
        trunk_height: usize,
        trunk_width: usize,
    ) -> Result<()> {
        for (height, width) in self.retained_stage_shapes(trunk_height, trunk_width)? {
            let _ = self.cached_position_encoding_base(
                device,
                dtype,
                self.config.d_model,
                height,
                width,
            )?;
        }
        Ok(())
    }

    pub fn forward(&self, trunk: &ViTDetTrunkOutput) -> Result<VisualBackboneOutput> {
        let Some(feature_map) = trunk.stage_features.last() else {
            candle::bail!("sam3 neck expects at least one trunk feature map")
        };
        let feature_map = feature_map.permute((0, 3, 1, 2))?;
        // cuDNN backward-data cannot consume the NHWC-as-NCHW view used as
        // input by the transposed-convolution stages. Materialize that shared
        // input once instead of once per stage and branch.
        #[cfg(feature = "cudnn")]
        let feature_map = {
            let _range = profiling::range("sam3.neck.shared_trunk_layout");
            feature_map.contiguous()?
        };
        let backbone_fpn = {
            let _range = profiling::range("sam3.neck.primary");
            self.forward_branch(&self.stages, &feature_map)?
        };
        let vision_pos_enc = {
            let _range = profiling::range("sam3.neck.position_encodings");
            self.build_position_encodings(&backbone_fpn, self.config.d_model)?
        };
        let tracker_sequences = Some(build_tracker_visual_sequences(
            backbone_fpn.as_slice(),
            vision_pos_enc.as_slice(),
        )?);
        let (sam2_backbone_fpn, sam2_pos_enc) = match &self.sam2_stages {
            Some(stages) => {
                let branch = {
                    let _range = profiling::range("sam3.neck.sam2");
                    self.forward_branch(stages, &feature_map)?
                };
                let pos: Vec<_> = vision_pos_enc.iter().map(Tensor::clone).collect();
                (Some(branch), Some(pos))
            }
            None => (None, None),
        };
        let tracker_sam2_sequences = match (&sam2_backbone_fpn, &tracker_sequences) {
            (Some(backbone_fpn), Some(primary_sequences)) => {
                Some(build_tracker_visual_sequences_reusing_position_layouts(
                    backbone_fpn.as_slice(),
                    primary_sequences,
                )?)
            }
            _ => None,
        };
        Ok(VisualBackboneOutput {
            backbone_fpn,
            vision_pos_enc,
            sam2_backbone_fpn,
            sam2_pos_enc,
            tracker_sequences,
            tracker_sam2_sequences,
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

    fn build_position_encodings(&self, features: &[Tensor], d_model: usize) -> Result<Vec<Tensor>> {
        let mut encodings = Vec::with_capacity(features.len());
        for feature in features {
            encodings.push(self.cached_2d_sine_position_encoding(feature, d_model)?);
        }
        Ok(encodings)
    }

    fn cached_2d_sine_position_encoding(&self, feature: &Tensor, d_model: usize) -> Result<Tensor> {
        let (batch_size, channels, height, width) = feature.dims4()?;
        if channels != d_model {
            candle::bail!("sam3 neck expected projected feature width {d_model}, got {channels}")
        }
        let base = self.cached_position_encoding_base(
            feature.device(),
            feature.dtype(),
            d_model,
            height,
            width,
        )?;
        if batch_size == 1 {
            Ok(base)
        } else {
            base.repeat((batch_size, 1, 1, 1))
        }
    }

    fn cached_position_encoding_base(
        &self,
        device: &Device,
        dtype: DType,
        d_model: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let key = PositionEncodingCacheKey {
            device: format!("{:?}", device),
            dtype: format!("{:?}", dtype),
            d_model,
            height,
            width,
        };
        let cached = {
            let cache = self
                .position_encoding_cache
                .lock()
                .expect("neck cache lock poisoned");
            cache.get(&key).cloned()
        };
        match cached {
            Some(tensor) => Ok(tensor),
            None => {
                let base = build_2d_sine_position_encoding_grid(
                    device,
                    dtype,
                    1,
                    d_model,
                    height,
                    width,
                    true,
                    2.0 * std::f32::consts::PI,
                    10_000f32,
                )?;
                let mut cache = self
                    .position_encoding_cache
                    .lock()
                    .expect("neck cache lock poisoned");
                Ok(cache.entry(key).or_insert_with(|| base.clone()).clone())
            }
        }
    }

    fn retained_stage_shapes(
        &self,
        trunk_height: usize,
        trunk_width: usize,
    ) -> Result<Vec<(usize, usize)>> {
        let mut shapes = self
            .stages
            .iter()
            .map(|stage| stage.output_shape(trunk_height, trunk_width))
            .collect::<Vec<_>>();
        if self.config.scalp > shapes.len() {
            candle::bail!(
                "sam3 neck scalp {} exceeds number of generated levels {}",
                self.config.scalp,
                shapes.len()
            )
        }
        shapes.truncate(shapes.len() - self.config.scalp);
        Ok(shapes)
    }
}

fn build_stages(
    config: &NeckConfig,
    branch: &str,
    vb: VarBuilder,
) -> Result<Vec<FeaturePyramidStage>> {
    let mut stages = Vec::with_capacity(config.scale_factors.len());
    for (index, scale_factor) in config.scale_factors.iter().copied().enumerate() {
        let kind = PyramidStageKind::from_scale_factor(scale_factor)?;
        let input_channels = stage_input_channels(kind, config.d_model)?;
        stages.push(FeaturePyramidStage::new(
            kind,
            branch,
            index,
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
