use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::EncoderConfig;
use super::geometry::EncodedPrompt;

#[derive(Debug)]
pub struct FusionEncoderOutput {
    pub memory: Tensor,
    pub pos_embed: Tensor,
    pub padding_mask: Tensor,
    pub level_start_index: Tensor,
    pub spatial_shapes: Tensor,
    pub valid_ratios: Tensor,
}

#[derive(Debug)]
pub struct Sam3FusionEncoder {
    config: EncoderConfig,
}

impl Sam3FusionEncoder {
    pub fn new(config: &EncoderConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn forward(
        &self,
        _visual_features: &[Tensor],
        _visual_pos_embeds: &[Tensor],
        _prompt: &EncodedPrompt,
    ) -> Result<FusionEncoderOutput> {
        candle::bail!("sam3 fusion encoder scaffold only: visual-language prompt fusion is not implemented yet")
    }
}
