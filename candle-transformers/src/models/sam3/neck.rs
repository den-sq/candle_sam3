use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::NeckConfig;
use super::vitdet::ViTDetTrunkOutput;

#[derive(Debug)]
pub struct VisualBackboneOutput {
    pub backbone_fpn: Vec<Tensor>,
    pub vision_pos_enc: Vec<Tensor>,
    pub sam2_backbone_fpn: Option<Vec<Tensor>>,
    pub sam2_pos_enc: Option<Vec<Tensor>>,
}

#[derive(Debug)]
pub struct Sam3DualViTDetNeck {
    config: NeckConfig,
}

impl Sam3DualViTDetNeck {
    pub fn new(config: &NeckConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &NeckConfig {
        &self.config
    }

    pub fn forward(&self, _trunk: &ViTDetTrunkOutput) -> Result<VisualBackboneOutput> {
        candle::bail!("sam3 dual neck scaffold only: simple-FPN projection and SAM2 side neck are not implemented yet")
    }
}
