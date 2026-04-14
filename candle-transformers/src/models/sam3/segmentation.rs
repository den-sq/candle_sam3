use candle::{DType, Result, Tensor};
use candle_nn::{
    group_norm, Conv2d, Conv2dConfig, GroupNorm, LayerNorm, Linear, Module, VarBuilder,
};

use super::config::SegmentationConfig;
use super::debug;
use super::decoder::DecoderOutput;

const SEGMENTATION_NUM_HEADS: usize = 8;
const GROUP_NORM_GROUPS: usize = 8;

#[derive(Debug)]
pub struct SegmentationOutput {
    pub mask_logits: Tensor,
    pub semantic_logits: Tensor,
    pub presence_logits: Option<Tensor>,
}

#[derive(Debug)]
struct SegmentationAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SegmentationAttention {
    fn new(hidden_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj_weight = vb.get((3 * hidden_dim, hidden_dim), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * hidden_dim, "in_proj_bias")?;
        let split_weights = in_proj_weight.chunk(3, 0)?;
        let split_biases = in_proj_bias.chunk(3, 0)?;
        let q_proj = Linear::new(split_weights[0].clone(), Some(split_biases[0].clone()));
        let k_proj = Linear::new(split_weights[1].clone(), Some(split_biases[1].clone()));
        let v_proj = Linear::new(split_weights[2].clone(), Some(split_biases[2].clone()));
        let out_proj = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("out_proj"))?;
        let head_dim = hidden_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn project(
        &self,
        xs: &Tensor,
        linear: &Linear,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        linear
            .forward(&xs.transpose(0, 1)?.contiguous()?)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch_size * self.num_heads, seq_len, self.head_dim))?
            .to_dtype(DType::F32)
    }

    fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        key_padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (tgt_len, batch_size, hidden_size) = query.dims3()?;
        let src_len = key_value.dim(0)?;
        let q = (self.project(query, &self.q_proj, batch_size, tgt_len)? * self.scale)?;
        let k = self.project(key_value, &self.k_proj, batch_size, src_len)?;
        let v = self.project(key_value, &self.v_proj, batch_size, src_len)?;
        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        if let Some(key_padding_mask) = key_padding_mask {
            let key_padding_mask = normalize_padding_mask(key_padding_mask, batch_size, src_len)?;
            let additive_mask = (key_padding_mask.to_dtype(DType::F32)? * -1e9f64)?
                .reshape((batch_size, 1, 1, src_len))?
                .repeat((1, self.num_heads, tgt_len, 1))?
                .reshape((batch_size * self.num_heads, tgt_len, src_len))?;
            attn = attn.broadcast_add(&additive_mask)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let hidden_states = attn
            .matmul(&v)?
            .reshape((batch_size, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch_size, tgt_len, hidden_size))?;
        self.out_proj
            .forward(&hidden_states)?
            .transpose(0, 1)?
            .contiguous()
    }
}

#[derive(Debug)]
struct MaskEmbedMlp {
    layers: Vec<Linear>,
}

impl MaskEmbedMlp {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let dims = [hidden_dim, hidden_dim, hidden_dim, hidden_dim];
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for layer_idx in 0..(dims.len() - 1) {
            layers.push(candle_nn::linear(
                dims[layer_idx],
                dims[layer_idx + 1],
                vb.pp("layers").pp(layer_idx),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden_states = xs.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states)?;
            if layer_idx + 1 != self.layers.len() {
                hidden_states = hidden_states.relu()?;
            }
        }
        Ok(hidden_states)
    }
}

#[derive(Debug)]
struct MaskPredictor {
    mask_embed: MaskEmbedMlp,
}

impl MaskPredictor {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            mask_embed: MaskEmbedMlp::new(hidden_dim, vb.pp("mask_embed"))?,
        })
    }

    fn forward(&self, obj_queries: &Tensor, pixel_embed: &Tensor) -> Result<Tensor> {
        let (batch_size, num_queries, hidden_dim) = obj_queries.dims3()?;
        let pixel_shape = pixel_embed.dims4()?;
        if pixel_shape.0 != batch_size || pixel_shape.1 != hidden_dim {
            candle::bail!(
                "sam3 segmentation mask predictor expected pixel embed shape ({batch_size}, {hidden_dim}, H, W), got {pixel_shape:?}"
            )
        }
        let (_, _, height, width) = pixel_shape;
        debug::capture_tensor("segmentation.mask_predictor.query_input", obj_queries)?;
        let query_embed = self.mask_embed.forward(obj_queries)?;
        debug::capture_tensor("segmentation.mask_predictor.query_embed", &query_embed)?;
        let pixel_embed = pixel_embed.reshape((batch_size, hidden_dim, height * width))?;
        debug::capture_tensor("segmentation.mask_predictor.pixel_flat", &pixel_embed)?;
        query_embed
            .matmul(&pixel_embed)?
            .reshape((batch_size, num_queries, height, width))
    }
}

#[derive(Debug)]
struct LinearPresenceHead {
    linear: Linear,
}

impl LinearPresenceHead {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(hidden_dim, 1, vb.pp("2"))?,
        })
    }

    fn forward(&self, pooled_hidden_states: &Tensor) -> Result<Tensor> {
        self.linear.forward(pooled_hidden_states)
    }
}

#[derive(Debug)]
struct PixelDecoder {
    conv_layers: Vec<Conv2d>,
    norms: Vec<GroupNorm>,
    out_dim: usize,
}

impl PixelDecoder {
    fn new(hidden_dim: usize, num_upsampling_stages: usize, vb: VarBuilder) -> Result<Self> {
        if hidden_dim % GROUP_NORM_GROUPS != 0 {
            candle::bail!(
                "sam3 segmentation hidden_dim ({hidden_dim}) must be divisible by group norm groups ({GROUP_NORM_GROUPS})"
            )
        }
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut conv_layers = Vec::with_capacity(num_upsampling_stages);
        let mut norms = Vec::with_capacity(num_upsampling_stages);
        for layer_idx in 0..num_upsampling_stages {
            conv_layers.push(candle_nn::conv2d(
                hidden_dim,
                hidden_dim,
                3,
                conv_cfg,
                vb.pp("conv_layers").pp(layer_idx),
            )?);
            norms.push(group_norm(
                GROUP_NORM_GROUPS,
                hidden_dim,
                1e-5,
                vb.pp("norms").pp(layer_idx),
            )?);
        }
        Ok(Self {
            conv_layers,
            norms,
            out_dim: hidden_dim,
        })
    }

    fn forward(&self, backbone_feats: &[Tensor]) -> Result<Tensor> {
        let Some(mut prev_fpn) = backbone_feats.last().cloned() else {
            candle::bail!("sam3 segmentation pixel decoder expects at least one feature level")
        };
        debug::capture_tensor("segmentation.pixel_decoder.initial_prev_fpn", &prev_fpn)?;
        for (layer_idx, curr_fpn) in backbone_feats[..backbone_feats.len().saturating_sub(1)]
            .iter()
            .rev()
            .enumerate()
        {
            if layer_idx >= self.conv_layers.len() || layer_idx >= self.norms.len() {
                candle::bail!(
                    "sam3 segmentation pixel decoder has {} conv stages but needs at least {}",
                    self.conv_layers.len(),
                    layer_idx + 1
                )
            }
            let curr_shape = curr_fpn.dims4()?;
            let prev_shape = prev_fpn.dims4()?;
            if curr_shape.0 != prev_shape.0 || curr_shape.1 != prev_shape.1 {
                candle::bail!(
                    "sam3 segmentation pixel decoder expected aligned FPN batch/channels, got {curr_shape:?} and {prev_shape:?}"
                )
            }
            debug::capture_tensor(
                &format!("segmentation.pixel_decoder.stage.{layer_idx}.curr_fpn"),
                curr_fpn,
            )?;
            let upsampled = prev_fpn.upsample_nearest2d(curr_shape.2, curr_shape.3)?;
            debug::capture_tensor(
                &format!("segmentation.pixel_decoder.stage.{layer_idx}.upsampled_prev_fpn"),
                &upsampled,
            )?;
            prev_fpn = curr_fpn.broadcast_add(&upsampled)?;
            debug::capture_tensor(
                &format!("segmentation.pixel_decoder.stage.{layer_idx}.sum"),
                &prev_fpn,
            )?;
            prev_fpn = self.conv_layers[layer_idx].forward(&prev_fpn)?;
            debug::capture_tensor(
                &format!("segmentation.pixel_decoder.stage.{layer_idx}.conv"),
                &prev_fpn,
            )?;
            prev_fpn = self.norms[layer_idx].forward(&prev_fpn)?.relu()?;
            debug::capture_tensor(
                &format!("segmentation.pixel_decoder.stage.{layer_idx}.output"),
                &prev_fpn,
            )?;
        }
        Ok(prev_fpn)
    }
}

#[derive(Debug)]
pub struct UniversalSegmentationHead {
    config: SegmentationConfig,
    pixel_decoder: PixelDecoder,
    mask_predictor: MaskPredictor,
    cross_attend_prompt: Option<SegmentationAttention>,
    cross_attn_norm: Option<LayerNorm>,
    semantic_seg_head: Conv2d,
    instance_seg_head: Conv2d,
    presence_head: Option<LinearPresenceHead>,
}

impl UniversalSegmentationHead {
    pub fn new(config: &SegmentationConfig, vb: VarBuilder) -> Result<Self> {
        let pixel_decoder = PixelDecoder::new(
            config.hidden_dim,
            config.upsampling_stages,
            vb.pp("pixel_decoder"),
        )?;
        let cross_attend_prompt = if vb.contains_tensor("cross_attend_prompt.in_proj_weight") {
            if config.hidden_dim % SEGMENTATION_NUM_HEADS != 0 {
                candle::bail!(
                    "sam3 segmentation hidden_dim ({}) must be divisible by cross-attention heads ({SEGMENTATION_NUM_HEADS})",
                    config.hidden_dim
                )
            }
            Some(SegmentationAttention::new(
                config.hidden_dim,
                SEGMENTATION_NUM_HEADS,
                vb.pp("cross_attend_prompt"),
            )?)
        } else {
            None
        };
        let cross_attn_norm = if cross_attend_prompt.is_some() {
            Some(candle_nn::layer_norm(
                config.hidden_dim,
                1e-5,
                vb.pp("cross_attn_norm"),
            )?)
        } else {
            None
        };
        let semantic_seg_head = candle_nn::conv2d(
            pixel_decoder.out_dim,
            1,
            1,
            Conv2dConfig::default(),
            vb.pp("semantic_seg_head"),
        )?;
        let instance_seg_head = candle_nn::conv2d(
            pixel_decoder.out_dim,
            config.hidden_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("instance_seg_head"),
        )?;
        let presence_head = if config.presence_head && vb.contains_tensor("presence_head.2.weight")
        {
            Some(LinearPresenceHead::new(
                config.hidden_dim,
                vb.pp("presence_head"),
            )?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            pixel_decoder,
            mask_predictor: MaskPredictor::new(config.hidden_dim, vb.pp("mask_predictor"))?,
            cross_attend_prompt,
            cross_attn_norm,
            semantic_seg_head,
            instance_seg_head,
            presence_head,
        })
    }

    pub fn config(&self) -> &SegmentationConfig {
        &self.config
    }

    pub fn forward(
        &self,
        backbone_fpn: &[Tensor],
        decoder_out: &DecoderOutput,
        encoder_hidden_states: &Tensor,
        prompt: Option<&Tensor>,
        prompt_mask: Option<&Tensor>,
    ) -> Result<SegmentationOutput> {
        let mut encoder_hidden_states = encoder_hidden_states.clone();
        debug::capture_tensor(
            "segmentation.encoder_hidden_states_input",
            &encoder_hidden_states,
        )?;
        if let (Some(cross_attend_prompt), Some(cross_attn_norm)) =
            (&self.cross_attend_prompt, &self.cross_attn_norm)
        {
            let prompt = prompt.ok_or_else(|| {
                candle::Error::Msg(
                    "sam3 segmentation head requires prompt features for prompt cross-attention"
                        .to_owned(),
                )
            })?;
            let prompt_mask = prompt_mask.ok_or_else(|| {
                candle::Error::Msg(
                    "sam3 segmentation head requires a prompt padding mask for prompt cross-attention"
                        .to_owned(),
                )
            })?;
            let normed_encoder = cross_attn_norm.forward(&encoder_hidden_states)?;
            debug::capture_tensor("segmentation.encoder_hidden_states_normed", &normed_encoder)?;
            let prompt_attn =
                cross_attend_prompt.forward(&normed_encoder, prompt, Some(prompt_mask))?;
            debug::capture_tensor("segmentation.prompt_attn", &prompt_attn)?;
            encoder_hidden_states = (prompt_attn + encoder_hidden_states)?;
            debug::capture_tensor(
                "segmentation.encoder_hidden_states_after_prompt",
                &encoder_hidden_states,
            )?;
        }

        let presence_logits = match &self.presence_head {
            Some(presence_head) => Some(presence_head.forward(&encoder_hidden_states.mean(0)?)?),
            None => None,
        };
        let pixel_embed = self.embed_pixels(backbone_fpn, &encoder_hidden_states)?;
        debug::capture_tensor("segmentation.pixel_embed", &pixel_embed)?;
        let instance_embeds = self.instance_seg_head.forward(&pixel_embed)?;
        debug::capture_tensor("segmentation.instance_embeds", &instance_embeds)?;
        let mask_logits = self
            .mask_predictor
            .forward(&decoder_out.queries, &instance_embeds)?;
        debug::capture_tensor("segmentation.mask_logits", &mask_logits)?;
        let semantic_logits = self.semantic_seg_head.forward(&pixel_embed)?;
        debug::capture_tensor("segmentation.semantic_logits", &semantic_logits)?;
        if let Some(presence_logits) = &presence_logits {
            debug::capture_tensor("segmentation.presence_logits", presence_logits)?;
        }
        Ok(SegmentationOutput {
            mask_logits,
            semantic_logits,
            presence_logits,
        })
    }

    fn embed_pixels(
        &self,
        backbone_fpn: &[Tensor],
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let Some(last_feature) = backbone_fpn.last() else {
            candle::bail!("sam3 segmentation head expects at least one backbone FPN level")
        };
        let (batch_size, channels, height, width) = last_feature.dims4()?;
        if channels != self.config.hidden_dim {
            candle::bail!(
                "sam3 segmentation head expected deepest FPN channels {}, got {channels}",
                self.config.hidden_dim
            )
        }
        let encoder_shape = encoder_hidden_states.dims3()?;
        if encoder_shape.1 != batch_size || encoder_shape.2 != channels {
            candle::bail!(
                "sam3 segmentation head expected encoder hidden states shape (seq, {batch_size}, {channels}), got {encoder_shape:?}"
            )
        }
        if encoder_shape.0 != height * width {
            candle::bail!(
                "sam3 segmentation head expected encoder sequence length {} to match deepest FPN area {}x{}, got {}",
                height * width,
                height,
                width,
                encoder_shape.0
            )
        }
        let mut backbone_visual_feats = backbone_fpn.iter().cloned().collect::<Vec<_>>();
        let last_idx = backbone_visual_feats.len() - 1;
        let encoder_visual_embed = encoder_hidden_states
            .permute((1, 2, 0))?
            .reshape((batch_size, channels, height, width))?;
        debug::capture_tensor("segmentation.encoder_visual_embed", &encoder_visual_embed)?;
        backbone_visual_feats[last_idx] = encoder_visual_embed;
        self.pixel_decoder.forward(&backbone_visual_feats)
    }
}

fn normalize_padding_mask(mask: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
    match mask.dims() {
        [b, s] if *b == batch_size && *s == seq_len => Ok(mask.clone()),
        [s, b] if *s == seq_len && *b == batch_size => Ok(mask.transpose(0, 1)?.contiguous()?),
        shape => candle::bail!(
            "sam3 segmentation expected padding mask shape ({batch_size}, {seq_len}) or ({seq_len}, {batch_size}), got {shape:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use candle::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;
    use serde::Deserialize;

    use super::{SegmentationOutput, UniversalSegmentationHead};
    use crate::models::sam3::config::SegmentationConfig;
    use crate::models::sam3::debug::{self, DebugExporter};
    use crate::models::sam3::decoder::DecoderOutput;

    #[derive(Debug, Deserialize)]
    struct SegmentationFixtureMetadata {
        hidden_dim: usize,
        upsampling_stages: usize,
        num_queries: usize,
    }

    #[test]
    fn segmentation_head_returns_mask_and_semantic_logits() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let vb =
            VarBuilder::from_tensors(segmentation_weights(&config, &device)?, DType::F32, &device);
        let head = UniversalSegmentationHead::new(&config, vb)?;
        let backbone_fpn = vec![
            Tensor::zeros((1, config.hidden_dim, 8, 8), DType::F32, &device)?,
            Tensor::zeros((1, config.hidden_dim, 4, 4), DType::F32, &device)?,
            Tensor::zeros((1, config.hidden_dim, 2, 2), DType::F32, &device)?,
        ];
        let decoder_out = DecoderOutput {
            queries: Tensor::zeros((1, 3, config.hidden_dim), DType::F32, &device)?,
            reference_boxes: Tensor::zeros((1, 3, 4), DType::F32, &device)?,
            pred_logits: Tensor::zeros((1, 3, 1), DType::F32, &device)?,
            pred_boxes: Tensor::zeros((1, 3, 4), DType::F32, &device)?,
            pred_boxes_xyxy: Tensor::zeros((1, 3, 4), DType::F32, &device)?,
            presence_logits: None,
        };
        let encoder_hidden_states = Tensor::zeros((4, 1, config.hidden_dim), DType::F32, &device)?;
        let prompt = Tensor::zeros((5, 1, config.hidden_dim), DType::F32, &device)?;
        let prompt_mask = Tensor::zeros((1, 5), DType::U8, &device)?;
        let output = head.forward(
            &backbone_fpn,
            &decoder_out,
            &encoder_hidden_states,
            Some(&prompt),
            Some(&prompt_mask),
        )?;
        assert_output(output, &config)
    }

    #[test]
    fn segmentation_fixture_smoke_final_matches_upstream() -> Result<()> {
        let output = run_fixture_segmentation_forward()?;
        let expected = load_segmentation_fixture_tensors("fixture.safetensors")?;
        assert_tensor_close(
            &output.mask_logits,
            fixture_tensor(&expected, "segmentation.mask_logits")?,
            1e-5,
            "segmentation.mask_logits",
        )?;
        assert_tensor_close(
            &output.semantic_logits,
            fixture_tensor(&expected, "segmentation.semantic_logits")?,
            1e-5,
            "segmentation.semantic_logits",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn segmentation_fixture_pixel_path_matches_upstream() -> Result<()> {
        let (_output, debug_tensors) = run_fixture_segmentation()?;
        let expected = load_segmentation_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "segmentation.encoder_hidden_states_input",
            "segmentation.encoder_hidden_states_normed",
            "segmentation.prompt_attn",
            "segmentation.encoder_hidden_states_after_prompt",
            "segmentation.encoder_visual_embed",
            "segmentation.pixel_decoder.initial_prev_fpn",
            "segmentation.pixel_decoder.stage.0.curr_fpn",
            "segmentation.pixel_decoder.stage.0.upsampled_prev_fpn",
            "segmentation.pixel_decoder.stage.0.sum",
            "segmentation.pixel_decoder.stage.0.conv",
            "segmentation.pixel_decoder.stage.0.output",
            "segmentation.pixel_decoder.stage.1.curr_fpn",
            "segmentation.pixel_decoder.stage.1.upsampled_prev_fpn",
            "segmentation.pixel_decoder.stage.1.sum",
            "segmentation.pixel_decoder.stage.1.conv",
            "segmentation.pixel_decoder.stage.1.output",
            "segmentation.pixel_embed",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn segmentation_fixture_mask_predictor_matches_upstream() -> Result<()> {
        let (_output, debug_tensors) = run_fixture_segmentation()?;
        let expected = load_segmentation_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "segmentation.instance_embeds",
            "segmentation.mask_predictor.query_input",
            "segmentation.mask_predictor.query_embed",
            "segmentation.mask_predictor.pixel_flat",
            "segmentation.mask_logits",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn segmentation_fixture_final_parity_matches_upstream() -> Result<()> {
        let (output, debug_tensors) = run_fixture_segmentation()?;
        let expected = load_segmentation_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "segmentation.pixel_embed",
            "segmentation.instance_embeds",
            "segmentation.mask_logits",
            "segmentation.semantic_logits",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)?;
        assert_tensor_close(
            &output.mask_logits,
            fixture_tensor(&expected, "segmentation.mask_logits")?,
            1e-5,
            "segmentation.mask_logits",
        )?;
        assert_tensor_close(
            &output.semantic_logits,
            fixture_tensor(&expected, "segmentation.semantic_logits")?,
            1e-5,
            "segmentation.semantic_logits",
        )?;
        Ok(())
    }

    fn assert_output(output: SegmentationOutput, config: &SegmentationConfig) -> Result<()> {
        assert_eq!(output.mask_logits.dims4()?, (1, 3, 8, 8));
        assert_eq!(output.semantic_logits.dims4()?, (1, 1, 8, 8));
        assert!(output.presence_logits.is_none());
        assert_eq!(output.mask_logits.dim(1)?, 3);
        assert_eq!(output.semantic_logits.dim(1)?, 1);
        assert_eq!(output.mask_logits.dim(2)?, 8);
        assert_eq!(config.hidden_dim, 8);
        Ok(())
    }

    fn test_config() -> SegmentationConfig {
        SegmentationConfig {
            enabled: true,
            hidden_dim: 8,
            upsampling_stages: 3,
            aux_masks: false,
            presence_head: false,
        }
    }

    fn segmentation_weights(
        config: &SegmentationConfig,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for layer_idx in 0..config.upsampling_stages {
            tensors.insert(
                format!("pixel_decoder.conv_layers.{layer_idx}.weight"),
                Tensor::zeros(
                    (config.hidden_dim, config.hidden_dim, 3, 3),
                    DType::F32,
                    device,
                )?,
            );
            tensors.insert(
                format!("pixel_decoder.conv_layers.{layer_idx}.bias"),
                Tensor::zeros(config.hidden_dim, DType::F32, device)?,
            );
            tensors.insert(
                format!("pixel_decoder.norms.{layer_idx}.weight"),
                Tensor::ones(config.hidden_dim, DType::F32, device)?,
            );
            tensors.insert(
                format!("pixel_decoder.norms.{layer_idx}.bias"),
                Tensor::zeros(config.hidden_dim, DType::F32, device)?,
            );
        }
        for layer_idx in 0..3 {
            tensors.insert(
                format!("mask_predictor.mask_embed.layers.{layer_idx}.weight"),
                Tensor::zeros((config.hidden_dim, config.hidden_dim), DType::F32, device)?,
            );
            tensors.insert(
                format!("mask_predictor.mask_embed.layers.{layer_idx}.bias"),
                Tensor::zeros(config.hidden_dim, DType::F32, device)?,
            );
        }
        tensors.insert(
            "cross_attend_prompt.in_proj_weight".into(),
            Tensor::zeros(
                (3 * config.hidden_dim, config.hidden_dim),
                DType::F32,
                device,
            )?,
        );
        tensors.insert(
            "cross_attend_prompt.in_proj_bias".into(),
            Tensor::zeros(3 * config.hidden_dim, DType::F32, device)?,
        );
        tensors.insert(
            "cross_attend_prompt.out_proj.weight".into(),
            Tensor::zeros((config.hidden_dim, config.hidden_dim), DType::F32, device)?,
        );
        tensors.insert(
            "cross_attend_prompt.out_proj.bias".into(),
            Tensor::zeros(config.hidden_dim, DType::F32, device)?,
        );
        tensors.insert(
            "cross_attn_norm.weight".into(),
            Tensor::ones(config.hidden_dim, DType::F32, device)?,
        );
        tensors.insert(
            "cross_attn_norm.bias".into(),
            Tensor::zeros(config.hidden_dim, DType::F32, device)?,
        );
        tensors.insert(
            "semantic_seg_head.weight".into(),
            Tensor::zeros((1, config.hidden_dim, 1, 1), DType::F32, device)?,
        );
        tensors.insert(
            "semantic_seg_head.bias".into(),
            Tensor::zeros(1, DType::F32, device)?,
        );
        tensors.insert(
            "instance_seg_head.weight".into(),
            Tensor::zeros(
                (config.hidden_dim, config.hidden_dim, 1, 1),
                DType::F32,
                device,
            )?,
        );
        tensors.insert(
            "instance_seg_head.bias".into(),
            Tensor::zeros(config.hidden_dim, DType::F32, device)?,
        );
        Ok(tensors)
    }

    fn fixture_metadata() -> Result<SegmentationFixtureMetadata> {
        let path = segmentation_fixture_dir().join("metadata.json");
        let contents = fs::read_to_string(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to read segmentation fixture metadata {}: {err}",
                path.display()
            ))
        })?;
        serde_json::from_str(&contents).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to parse segmentation fixture metadata {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_config() -> Result<SegmentationConfig> {
        let metadata = fixture_metadata()?;
        Ok(SegmentationConfig {
            enabled: true,
            hidden_dim: metadata.hidden_dim,
            upsampling_stages: metadata.upsampling_stages,
            aux_masks: false,
            presence_head: false,
        })
    }

    fn segmentation_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_segmentation_unit")
    }

    fn load_segmentation_fixture_tensors(file_name: &str) -> Result<HashMap<String, Tensor>> {
        let path = segmentation_fixture_dir().join(file_name);
        candle::safetensors::load(&path, &Device::Cpu).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load segmentation fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_tensor<'a>(fixture: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
        fixture.get(key).ok_or_else(|| {
            candle::Error::Msg(format!("segmentation fixture is missing tensor `{key}`"))
        })
    }

    fn unique_temp_dir(label: &str) -> Result<PathBuf> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("system clock error: {err}")))?;
        let path = std::env::temp_dir().join(format!(
            "sam3_segmentation_fixture_{label}_{}_{}",
            std::process::id(),
            stamp.as_nanos()
        ));
        fs::create_dir_all(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create temp debug dir {}: {err}",
                path.display()
            ))
        })?;
        Ok(path)
    }

    fn fixture_backbone_fpn(fixture: &HashMap<String, Tensor>) -> Vec<Tensor> {
        let mut levels = Vec::new();
        for idx in 0.. {
            let key = format!("inputs/backbone_fpn.{idx}");
            let Some(level) = fixture.get(&key) else {
                break;
            };
            levels.push(level.clone());
        }
        levels
    }

    fn run_fixture_segmentation() -> Result<(SegmentationOutput, HashMap<String, Tensor>)> {
        let device = Device::Cpu;
        let config = fixture_config()?;
        let weights = load_segmentation_fixture_tensors("segmentation_weights.safetensors")?;
        let fixture = load_segmentation_fixture_tensors("fixture.safetensors")?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let head = UniversalSegmentationHead::new(&config, vb)?;
        let backbone_fpn = fixture_backbone_fpn(&fixture);
        if backbone_fpn.is_empty() {
            candle::bail!("segmentation fixture did not include any backbone FPN levels");
        }
        let num_queries = fixture_metadata()?.num_queries;
        let decoder_out = DecoderOutput {
            queries: fixture_tensor(&fixture, "inputs/decoder_queries")?.clone(),
            reference_boxes: Tensor::zeros((1, num_queries, 4), DType::F32, &device)?,
            pred_logits: Tensor::zeros((1, num_queries, 1), DType::F32, &device)?,
            pred_boxes: Tensor::zeros((1, num_queries, 4), DType::F32, &device)?,
            pred_boxes_xyxy: Tensor::zeros((1, num_queries, 4), DType::F32, &device)?,
            presence_logits: None,
        };
        let encoder_hidden_states =
            fixture_tensor(&fixture, "inputs/encoder_hidden_states")?.clone();
        let prompt = fixture_tensor(&fixture, "inputs/prompt")?.clone();
        let prompt_mask = fixture_tensor(&fixture, "inputs/prompt_mask")?.clone();

        let debug_dir = unique_temp_dir("forward")?;
        debug::set_exporter(Some(DebugExporter::new(&debug_dir)?));
        let output = head.forward(
            &backbone_fpn,
            &decoder_out,
            &encoder_hidden_states,
            Some(&prompt),
            Some(&prompt_mask),
        )?;
        debug::finish()?;
        let debug_tensors =
            candle::safetensors::load(debug_dir.join("debug_tensors.safetensors"), &device)
                .map_err(|err| {
                    candle::Error::Msg(format!(
                        "failed to load segmentation debug tensors from {}: {err}",
                        debug_dir.display()
                    ))
                })?;
        let _ = fs::remove_dir_all(&debug_dir);
        Ok((output, debug_tensors))
    }

    fn run_fixture_segmentation_forward() -> Result<SegmentationOutput> {
        let device = Device::Cpu;
        let config = fixture_config()?;
        let weights = load_segmentation_fixture_tensors("segmentation_weights.safetensors")?;
        let fixture = load_segmentation_fixture_tensors("fixture.safetensors")?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let head = UniversalSegmentationHead::new(&config, vb)?;
        let backbone_fpn = fixture_backbone_fpn(&fixture);
        if backbone_fpn.is_empty() {
            candle::bail!("segmentation fixture did not include any backbone FPN levels");
        }
        let num_queries = fixture_metadata()?.num_queries;
        let decoder_out = DecoderOutput {
            queries: fixture_tensor(&fixture, "inputs/decoder_queries")?.clone(),
            reference_boxes: Tensor::zeros((1, num_queries, 4), DType::F32, &device)?,
            pred_logits: Tensor::zeros((1, num_queries, 1), DType::F32, &device)?,
            pred_boxes: Tensor::zeros((1, num_queries, 4), DType::F32, &device)?,
            pred_boxes_xyxy: Tensor::zeros((1, num_queries, 4), DType::F32, &device)?,
            presence_logits: None,
        };
        let encoder_hidden_states =
            fixture_tensor(&fixture, "inputs/encoder_hidden_states")?.clone();
        let prompt = fixture_tensor(&fixture, "inputs/prompt")?.clone();
        let prompt_mask = fixture_tensor(&fixture, "inputs/prompt_mask")?.clone();
        head.forward(
            &backbone_fpn,
            &decoder_out,
            &encoder_hidden_states,
            Some(&prompt),
            Some(&prompt_mask),
        )
    }

    fn assert_debug_keys_close(
        actual: &HashMap<String, Tensor>,
        expected: &HashMap<String, Tensor>,
        keys: &[&str],
        atol: f32,
    ) -> Result<()> {
        let mut failures = Vec::new();
        for key in keys {
            let Some(actual_tensor) = actual.get(*key) else {
                failures.push(format!("{key}: missing from Candle debug output"));
                continue;
            };
            let Some(expected_tensor) = expected.get(*key) else {
                failures.push(format!("{key}: missing from fixture"));
                continue;
            };
            if let Err(err) = assert_tensor_close(actual_tensor, expected_tensor, atol, key) {
                failures.push(err.to_string());
            }
        }
        if failures.is_empty() {
            return Ok(());
        }
        candle::bail!("{}", failures.join("\n"));
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
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        let max_abs_diff = actual
            .broadcast_sub(&expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        if max_abs_diff > atol {
            candle::bail!("{name}: max_abs_diff={max_abs_diff:.8}");
        }
        Ok(())
    }
}
