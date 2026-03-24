use std::path::Path;

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::checkpoint::Sam3CheckpointSource;
use super::config::Config;
use super::decoder::Sam3TransformerDecoder;
use super::encoder::Sam3FusionEncoder;
use super::geometry::{GeometryPrompt, SequenceGeometryEncoder};
use super::neck::Sam3DualViTDetNeck;
use super::segmentation::UniversalSegmentationHead;
use super::text::{Sam3TextEncoder, TextEncoding};
use super::vitdet::Sam3ViTDetTrunk;

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
        let geometry =
            SequenceGeometryEncoder::new(&config.geometry, vb.pp("input_geometry_encoder"))?;
        let encoder = Sam3FusionEncoder::new(&config.encoder, vb.pp("transformer").pp("encoder"))?;
        let decoder =
            Sam3TransformerDecoder::new(&config.decoder, vb.pp("transformer").pp("decoder"))?;
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

#[cfg(test)]
mod tests {
    use super::{GroundingOutput, ImageSize, Sam3ImageState};
    use candle::{Device, Result, Tensor};

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
}
