use std::path::Path;

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::checkpoint::Sam3CheckpointSource;
use super::config::Config;
use super::decoder::{DecoderOutput, Sam3TransformerDecoder};
use super::encoder::{FusionEncoderOutput, Sam3FusionEncoder};
use super::geometry::{EncodedPrompt, GeometryPrompt, SequenceGeometryEncoder};
use super::neck::{Sam3DualViTDetNeck, VisualBackboneOutput};
use super::segmentation::{SegmentationOutput, UniversalSegmentationHead};
use super::text::{Sam3TextEncoder, TextEncoding};
use super::vitdet::{Sam3ViTDetTrunk, ViTDetTrunkOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub height: usize,
    pub width: usize,
}

impl ImageSize {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub const fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Sam3PromptState {
    pub text_prompt: Option<String>,
    pub geometry_prompt: GeometryPrompt,
}

impl Sam3PromptState {
    pub fn with_text_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.text_prompt = Some(prompt.into());
        self
    }

    pub fn with_geometry_prompt(mut self, prompt: GeometryPrompt) -> Self {
        self.geometry_prompt = prompt;
        self
    }

    pub fn clear(mut self) -> Self {
        self.text_prompt = None;
        self.geometry_prompt = GeometryPrompt::default();
        self
    }
}

#[derive(Debug, Clone)]
pub struct Sam3ImageState {
    pub original_size: ImageSize,
    pub model_input_size: ImageSize,
    pub prompts: Sam3PromptState,
    pub last_output: Option<GroundingOutput>,
}

impl Sam3ImageState {
    pub fn new(original_size: ImageSize, model_input_size: ImageSize) -> Self {
        Self {
            original_size,
            model_input_size,
            prompts: Sam3PromptState::default(),
            last_output: None,
        }
    }

    pub fn text_prompt(&self) -> Option<&str> {
        self.prompts.text_prompt.as_deref()
    }

    pub fn geometry_prompt(&self) -> &GeometryPrompt {
        &self.prompts.geometry_prompt
    }

    pub fn with_text_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompts = self.prompts.with_text_prompt(prompt);
        self
    }

    pub fn with_geometry_prompt(mut self, prompt: GeometryPrompt) -> Self {
        self.prompts = self.prompts.with_geometry_prompt(prompt);
        self
    }

    pub fn clear_prompts(mut self) -> Self {
        self.prompts = self.prompts.clear();
        self
    }

    pub fn with_last_output(mut self, output: GroundingOutput) -> Self {
        self.last_output = Some(output);
        self
    }
}

#[derive(Debug, Clone)]
pub struct GroundingOutput {
    pub mask_logits: Tensor,
    pub masks: Tensor,
    pub boxes_xyxy: Tensor,
    pub scores: Tensor,
    pub presence_scores: Option<Tensor>,
}

#[derive(Debug)]
pub struct Sam3ImageModel {
    config: Config,
    vision_trunk: Sam3ViTDetTrunk,
    vision_neck: Sam3DualViTDetNeck,
    text: Sam3TextEncoder,
    geometry: SequenceGeometryEncoder,
    encoder: Sam3FusionEncoder,
    decoder: Sam3TransformerDecoder,
    segmentation: Option<UniversalSegmentationHead>,
}

impl Sam3ImageModel {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let vision_trunk = Sam3ViTDetTrunk::new(
            &config.vision,
            vb.pp("backbone").pp("vision_backbone").pp("trunk"),
        )?;
        let vision_neck =
            Sam3DualViTDetNeck::new(&config.neck, vb.pp("backbone").pp("vision_backbone"))?;
        let text = Sam3TextEncoder::new(&config.text, vb.pp("backbone").pp("language_backbone"))?;
        let geometry = SequenceGeometryEncoder::new(&config.geometry, vb.pp("geometry_encoder"))?;
        let encoder = Sam3FusionEncoder::new(&config.encoder, vb.pp("transformer").pp("encoder"))?;
        let decoder = Sam3TransformerDecoder::new(
            &config.decoder,
            vb.pp("transformer").pp("decoder"),
            vb.pp("dot_prod_scoring"),
        )?;
        let segmentation = if config.segmentation.enabled {
            Some(UniversalSegmentationHead::new(
                &config.segmentation,
                vb.pp("segmentation_head"),
            )?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            vision_trunk,
            vision_neck,
            text,
            geometry,
            encoder,
            decoder,
            segmentation,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn input_size(&self) -> ImageSize {
        ImageSize::square(self.config.image.image_size)
    }

    pub fn from_checkpoint_source(
        config: &Config,
        checkpoint: &Sam3CheckpointSource,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        checkpoint.load_image_model(config, dtype, device)
    }

    pub fn from_upstream_pth<P: AsRef<Path>>(
        config: &Config,
        checkpoint: P,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Sam3CheckpointSource::upstream_pth(checkpoint.as_ref().to_path_buf())
            .load_image_model(config, dtype, device)
    }

    pub fn set_image(&self, image: &Tensor) -> Result<Sam3ImageState> {
        let original_size = match image.rank() {
            3 => {
                let (_c, h, w) = image.dims3()?;
                ImageSize::new(h, w)
            }
            4 => {
                let (_b, _c, h, w) = image.dims4()?;
                ImageSize::new(h, w)
            }
            rank => candle::bail!("expected CHW or BCHW image tensor, got rank {rank}"),
        };
        Ok(Sam3ImageState::new(original_size, self.input_size()))
    }

    pub fn encode_text_tokens(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<TextEncoding> {
        self.text.forward(input_ids, attention_mask)
    }

    pub fn encode_image_features(&self, image: &Tensor) -> Result<VisualBackboneOutput> {
        let image = match image.rank() {
            3 => image.unsqueeze(0)?,
            4 => image.clone(),
            rank => candle::bail!("sam3 image encoder expects CHW or BCHW input, got rank {rank}"),
        };
        let trunk = self.vision_trunk.forward(&image)?;
        self.vision_neck.forward(&trunk)
    }

    pub fn encode_image_trunk(&self, image: &Tensor) -> Result<ViTDetTrunkOutput> {
        let image = match image.rank() {
            3 => image.unsqueeze(0)?,
            4 => image.clone(),
            rank => candle::bail!("sam3 image encoder expects CHW or BCHW input, got rank {rank}"),
        };
        self.vision_trunk.forward(&image)
    }

    pub fn encode_image_trunk_with_block_outputs(
        &self,
        image: &Tensor,
    ) -> Result<(ViTDetTrunkOutput, Vec<Tensor>)> {
        let image = match image.rank() {
            3 => image.unsqueeze(0)?,
            4 => image.clone(),
            rank => candle::bail!("sam3 image encoder expects CHW or BCHW input, got rank {rank}"),
        };
        self.vision_trunk.forward_with_block_outputs(&image)
    }

    pub fn encode_image_trunk_with_debug_blocks(
        &self,
        image: &Tensor,
        debug_blocks: &[usize],
    ) -> Result<(
        ViTDetTrunkOutput,
        Vec<Tensor>,
        std::collections::BTreeMap<String, Tensor>,
    )> {
        let image = match image.rank() {
            3 => image.unsqueeze(0)?,
            4 => image.clone(),
            rank => candle::bail!("sam3 image encoder expects CHW or BCHW input, got rank {rank}"),
        };
        self.vision_trunk
            .forward_with_debug_blocks(&image, debug_blocks)
    }

    pub fn encode_geometry_prompt(
        &self,
        prompt: &GeometryPrompt,
        visual_features: &VisualBackboneOutput,
    ) -> Result<EncodedPrompt> {
        self.geometry.encode(
            prompt,
            &visual_features.backbone_fpn,
            &visual_features.vision_pos_enc,
        )
    }

    pub fn encode_fused_prompt(
        &self,
        visual_features: &VisualBackboneOutput,
        prompt: &EncodedPrompt,
    ) -> Result<FusionEncoderOutput> {
        self.encoder.forward(
            &visual_features.backbone_fpn,
            &visual_features.vision_pos_enc,
            prompt,
        )
    }

    pub fn encode_fused_text(
        &self,
        visual_features: &VisualBackboneOutput,
        text: &TextEncoding,
    ) -> Result<FusionEncoderOutput> {
        self.encode_fused_prompt(visual_features, &encoded_prompt_from_text(text))
    }

    pub fn decode_grounding(
        &self,
        encoder_out: &FusionEncoderOutput,
        prompt: &EncodedPrompt,
    ) -> Result<DecoderOutput> {
        self.decoder
            .forward(encoder_out, &prompt.features, &prompt.padding_mask)
    }

    pub fn decode_text_grounding(
        &self,
        encoder_out: &FusionEncoderOutput,
        text: &TextEncoding,
    ) -> Result<DecoderOutput> {
        let prompt = encoded_prompt_from_text(text);
        self.decode_grounding(encoder_out, &prompt)
    }

    pub fn text_detection_scores(&self, decoder_out: &DecoderOutput) -> Result<Tensor> {
        let class_scores = decoder_out.pred_logits.apply(&candle_nn::ops::sigmoid)?;
        match &decoder_out.presence_logits {
            Some(presence_logits) => {
                let batch_size = presence_logits.dim(0)?;
                let presence_scores = presence_logits
                    .apply(&candle_nn::ops::sigmoid)?
                    .reshape((batch_size, 1, 1))?;
                class_scores.broadcast_mul(&presence_scores)
            }
            None => Ok(class_scores),
        }
    }

    pub fn segment_grounding(
        &self,
        visual_features: &VisualBackboneOutput,
        decoder_out: &DecoderOutput,
        encoder_out: &FusionEncoderOutput,
        prompt: &EncodedPrompt,
    ) -> Result<SegmentationOutput> {
        let segmentation = self.segmentation.as_ref().ok_or_else(|| {
            candle::Error::Msg("sam3 segmentation head is disabled in this config".to_owned())
        })?;
        segmentation.forward(
            &visual_features.backbone_fpn,
            decoder_out,
            &encoder_out.memory,
            Some(&prompt.features),
            Some(&prompt.padding_mask),
        )
    }

    pub fn segment_text_grounding(
        &self,
        visual_features: &VisualBackboneOutput,
        decoder_out: &DecoderOutput,
        encoder_out: &FusionEncoderOutput,
        text: &TextEncoding,
    ) -> Result<SegmentationOutput> {
        let prompt = encoded_prompt_from_text(text);
        self.segment_grounding(visual_features, decoder_out, encoder_out, &prompt)
    }

    pub fn ground_text(&self, state: &Sam3ImageState) -> Result<GroundingOutput> {
        let Some(_prompt) = state.text_prompt() else {
            candle::bail!("sam3 image state has no text prompt; call `with_text_prompt` first")
        };
        let _ = (
            &self.vision_trunk,
            &self.vision_neck,
            &self.text,
            &self.geometry,
            &self.encoder,
            &self.decoder,
            &self.segmentation,
        );
        candle::bail!(
            "sam3 image grounding scaffold only: image/text grounding pipeline not implemented yet"
        )
    }

    pub fn ground_geometry(&self, state: &Sam3ImageState) -> Result<GroundingOutput> {
        if state.geometry_prompt().is_empty() {
            candle::bail!(
                "sam3 image state has no geometry prompt; call `with_geometry_prompt` first"
            )
        }
        let _ = (
            &self.vision_trunk,
            &self.vision_neck,
            &self.text,
            &self.geometry,
            &self.encoder,
            &self.decoder,
            &self.segmentation,
        );
        candle::bail!("sam3 geometry grounding scaffold only: visual prompt and geometric prompt pipeline not implemented yet")
    }

    pub fn scaffold_milestones() -> [&'static str; 4] {
        [
            "image grounding parity against upstream detector",
            "checkpoint namespace/weight remapping validation",
            "interactive SAM task refinement on images",
            "video detector/tracker integration",
        ]
    }
}

fn encoded_prompt_from_text(text: &TextEncoding) -> EncodedPrompt {
    EncodedPrompt {
        features: text.memory.clone(),
        padding_mask: text.attention_mask.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{GroundingOutput, ImageSize, Sam3ImageState};
    use candle::{Device, Result, Tensor};
    use candle_nn::VarBuilder;

    use crate::models::sam3::Sam3ImageModel;
    use crate::models::sam3::{
        Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
        SegmentationConfig, TextConfig, VisionConfig,
    };

    #[test]
    fn image_state_tracks_prompt_state() {
        let state = Sam3ImageState::new(ImageSize::new(400, 600), ImageSize::square(1008))
            .with_text_prompt("player in white")
            .clear_prompts();
        assert_eq!(state.original_size, ImageSize::new(400, 600));
        assert_eq!(state.model_input_size, ImageSize::square(1008));
        assert_eq!(state.text_prompt(), None);
        assert!(state.geometry_prompt().is_empty());
    }

    #[test]
    fn image_state_can_store_last_output() -> Result<()> {
        let dev = Device::Cpu;
        let output = GroundingOutput {
            mask_logits: Tensor::zeros((1, 1, 2, 2), candle::DType::F32, &dev)?,
            masks: Tensor::zeros((1, 1, 2, 2), candle::DType::U8, &dev)?,
            boxes_xyxy: Tensor::zeros((1, 4), candle::DType::F32, &dev)?,
            scores: Tensor::zeros((1,), candle::DType::F32, &dev)?,
            presence_scores: None,
        };
        let state = Sam3ImageState::new(ImageSize::new(10, 12), ImageSize::square(1008))
            .with_last_output(output);
        assert!(state.last_output.is_some());
        Ok(())
    }

    #[test]
    fn image_model_can_run_text_fusion_and_decoder_stages() -> Result<()> {
        let dev = Device::Cpu;
        let config = tiny_config();
        let model = Sam3ImageModel::new(&config, VarBuilder::zeros(candle::DType::F32, &dev))?;
        let image = Tensor::zeros(
            (1, 3, config.image.image_size, config.image.image_size),
            candle::DType::F32,
            &dev,
        )?;
        let input_ids = Tensor::new(&[[1u32, 2, 3, 0]], &dev)?;
        let attention_mask = Tensor::new(&[[1u8, 1, 1, 0]], &dev)?;
        let text = model.encode_text_tokens(&input_ids, &attention_mask)?;
        let visual = model.encode_image_features(&image)?;
        let fused = model.encode_fused_text(&visual, &text)?;
        let decoder = model.decode_text_grounding(&fused, &text)?;
        let scores = model.text_detection_scores(&decoder)?;

        assert_eq!(visual.backbone_fpn.len(), 1);
        assert_eq!(fused.memory.dims3()?, (16, 1, config.encoder.d_model));
        assert_eq!(
            decoder.queries.dims3()?,
            (1, config.decoder.num_queries, config.decoder.d_model)
        );
        assert_eq!(
            decoder.pred_logits.dims3()?,
            (1, config.decoder.num_queries, 1)
        );
        assert_eq!(scores.dims3()?, (1, config.decoder.num_queries, 1));
        Ok(())
    }

    #[test]
    fn image_model_can_run_text_segmentation_stage() -> Result<()> {
        let dev = Device::Cpu;
        let config = tiny_segmentation_config();
        let model = Sam3ImageModel::new(&config, VarBuilder::zeros(candle::DType::F32, &dev))?;
        let image = Tensor::zeros(
            (1, 3, config.image.image_size, config.image.image_size),
            candle::DType::F32,
            &dev,
        )?;
        let input_ids = Tensor::new(&[[1u32, 2, 3, 0]], &dev)?;
        let attention_mask = Tensor::new(&[[1u8, 1, 1, 0]], &dev)?;
        let text = model.encode_text_tokens(&input_ids, &attention_mask)?;
        let visual = model.encode_image_features(&image)?;
        let fused = model.encode_fused_text(&visual, &text)?;
        let decoder = model.decode_text_grounding(&fused, &text)?;
        let segmentation = model.segment_text_grounding(&visual, &decoder, &fused, &text)?;

        assert_eq!(
            segmentation.mask_logits.dims4()?,
            (1, config.decoder.num_queries, 16, 16)
        );
        assert_eq!(segmentation.semantic_logits.dims4()?, (1, 1, 16, 16));
        Ok(())
    }

    fn tiny_config() -> Config {
        Config {
            image: ImageConfig {
                image_size: 56,
                image_mean: [0.5, 0.5, 0.5],
                image_std: [0.5, 0.5, 0.5],
            },
            vision: VisionConfig {
                image_size: 56,
                pretrain_image_size: 28,
                patch_size: 14,
                embed_dim: 16,
                depth: 0,
                num_heads: 2,
                mlp_ratio: 4.0,
                window_size: 2,
                global_attn_blocks: vec![],
                use_abs_pos: true,
                tile_abs_pos: true,
                use_rope: true,
                use_interp_rope: true,
                rope_theta: 10_000.0,
                rope_pt_size: 24,
                retain_cls_token: false,
                ln_pre: false,
            },
            text: TextConfig {
                d_model: 4,
                width: 8,
                heads: 2,
                layers: 1,
                context_length: 4,
                vocab_size: 16,
            },
            neck: NeckConfig {
                d_model: 4,
                scale_factors: [1.0, 0.5, 0.5, 0.5],
                scalp: 3,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 4,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 8,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 4,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 8,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 4,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 8,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: false,
                hidden_dim: 4,
                upsampling_stages: 1,
                aux_masks: false,
                presence_head: false,
            },
        }
    }

    fn tiny_segmentation_config() -> Config {
        Config {
            image: ImageConfig {
                image_size: 56,
                image_mean: [0.5, 0.5, 0.5],
                image_std: [0.5, 0.5, 0.5],
            },
            vision: VisionConfig {
                image_size: 56,
                pretrain_image_size: 28,
                patch_size: 14,
                embed_dim: 32,
                depth: 0,
                num_heads: 4,
                mlp_ratio: 4.0,
                window_size: 2,
                global_attn_blocks: vec![],
                use_abs_pos: true,
                tile_abs_pos: true,
                use_rope: true,
                use_interp_rope: true,
                rope_theta: 10_000.0,
                rope_pt_size: 24,
                retain_cls_token: false,
                ln_pre: false,
            },
            text: TextConfig {
                d_model: 8,
                width: 16,
                heads: 2,
                layers: 1,
                context_length: 4,
                vocab_size: 16,
            },
            neck: NeckConfig {
                d_model: 8,
                scale_factors: [4.0, 2.0, 1.0, 0.5],
                scalp: 1,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 8,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 16,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 8,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 16,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 8,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 16,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: true,
                hidden_dim: 8,
                upsampling_stages: 3,
                aux_masks: false,
                presence_head: false,
            },
        }
    }
}
