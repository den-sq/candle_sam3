use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::DecoderConfig;
use super::encoder::FusionEncoderOutput;

#[derive(Debug)]
pub struct DecoderOutput {
    pub queries: Tensor,
    pub reference_boxes: Tensor,
    pub pred_logits: Tensor,
    pub pred_boxes: Tensor,
    pub pred_boxes_xyxy: Tensor,
    pub presence_logits: Option<Tensor>,
}

#[derive(Debug)]
pub struct Sam3TransformerDecoder {
    config: DecoderConfig,
}

impl Sam3TransformerDecoder {
    pub fn new(config: &DecoderConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn forward(
        &self,
        _encoder_out: &FusionEncoderOutput,
        _prompt_features: &Tensor,
        _prompt_mask: &Tensor,
    ) -> Result<DecoderOutput> {
        candle::bail!("sam3 decoder scaffold only: DETR-style query decoding and presence-token scoring are not implemented yet")
    }
}
