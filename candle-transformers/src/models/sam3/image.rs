use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::Config;
use super::decoder::Sam3TransformerDecoder;
use super::encoder::Sam3FusionEncoder;
use super::geometry::{GeometryPrompt, SequenceGeometryEncoder};
use super::neck::Sam3DualViTDetNeck;
use super::segmentation::UniversalSegmentationHead;
use super::text::Sam3TextEncoder;
use super::vitdet::Sam3ViTDetTrunk;

#[derive(Debug)]
pub struct Sam3ImageState {
    pub original_height: usize,
    pub original_width: usize,
}

#[derive(Debug)]
pub struct GroundingOutput {
    pub mask_logits: Tensor,
    pub masks: Tensor,
    pub boxes_xyxy: Tensor,
    pub scores: Tensor,
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
        let vision_trunk =
            Sam3ViTDetTrunk::new(&config.vision, vb.pp("backbone").pp("vision_trunk"))?;
        let vision_neck =
            Sam3DualViTDetNeck::new(&config.neck, vb.pp("backbone").pp("vision_neck"))?;
        let text = Sam3TextEncoder::new(&config.text, vb.pp("backbone").pp("text_encoder"))?;
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

    pub fn set_image(&self, image: &Tensor) -> Result<Sam3ImageState> {
        let (original_height, original_width) = match image.rank() {
            3 => {
                let (_c, h, w) = image.dims3()?;
                (h, w)
            }
            4 => {
                let (_b, _c, h, w) = image.dims4()?;
                (h, w)
            }
            rank => candle::bail!("expected CHW or BCHW image tensor, got rank {rank}"),
        };
        Ok(Sam3ImageState {
            original_height,
            original_width,
        })
    }

    pub fn ground_text(&self, _state: &Sam3ImageState, _prompt: &str) -> Result<GroundingOutput> {
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

    pub fn ground_geometry(
        &self,
        _state: &Sam3ImageState,
        _prompt: &GeometryPrompt,
    ) -> Result<GroundingOutput> {
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
