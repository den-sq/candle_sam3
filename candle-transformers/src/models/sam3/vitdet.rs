use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::VisionConfig;

#[derive(Debug)]
pub struct ViTDetTrunkOutput {
    pub stage_features: Vec<Tensor>,
}

#[derive(Debug)]
pub struct Sam3ViTDetTrunk {
    config: VisionConfig,
}

impl Sam3ViTDetTrunk {
    pub fn new(config: &VisionConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn forward(&self, _images: &Tensor) -> Result<ViTDetTrunkOutput> {
        candle::bail!("sam3 ViTDet trunk scaffold only: window/global attention, abs-pos tiling, and 2D RoPE are not implemented yet")
    }
}
