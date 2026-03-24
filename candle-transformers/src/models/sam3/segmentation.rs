use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::SegmentationConfig;
use super::decoder::DecoderOutput;

#[derive(Debug)]
pub struct SegmentationOutput {
    pub mask_logits: Tensor,
    pub presence_logits: Option<Tensor>,
}

#[derive(Debug)]
pub struct UniversalSegmentationHead {
    config: SegmentationConfig,
}

impl UniversalSegmentationHead {
    pub fn new(config: &SegmentationConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &SegmentationConfig {
        &self.config
    }

    pub fn forward(
        &self,
        _backbone_fpn: &[Tensor],
        _decoder_out: &DecoderOutput,
        _encoder_hidden_states: &Tensor,
    ) -> Result<SegmentationOutput> {
        candle::bail!("sam3 segmentation head scaffold only: pixel decoder + mask prediction are not implemented yet")
    }
}
