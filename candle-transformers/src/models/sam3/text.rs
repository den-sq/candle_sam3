use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::TextConfig;

#[derive(Debug)]
pub struct TextEncoding {
    pub attention_mask: Tensor,
    pub memory: Tensor,
    pub input_embeddings: Tensor,
}

#[derive(Debug)]
pub struct Sam3TextEncoder {
    config: TextConfig,
}

impl Sam3TextEncoder {
    pub fn new(config: &TextConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    pub fn forward(&self, _input_ids: &Tensor, _attention_mask: &Tensor) -> Result<TextEncoding> {
        candle::bail!("sam3 text encoder scaffold only: token embedding + transformer + resize projection not implemented yet")
    }
}
