use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::GeometryConfig;

#[derive(Debug, Default, Clone)]
pub struct GeometryPrompt {
    pub boxes_cxcywh: Option<Tensor>,
    pub box_labels: Option<Tensor>,
    pub points_xy: Option<Tensor>,
    pub point_labels: Option<Tensor>,
    pub masks: Option<Tensor>,
    pub mask_labels: Option<Tensor>,
}

impl GeometryPrompt {
    pub fn is_empty(&self) -> bool {
        self.boxes_cxcywh.is_none()
            && self.box_labels.is_none()
            && self.points_xy.is_none()
            && self.point_labels.is_none()
            && self.masks.is_none()
            && self.mask_labels.is_none()
    }
}

#[derive(Debug)]
pub struct EncodedPrompt {
    pub features: Tensor,
    pub padding_mask: Tensor,
}

#[derive(Debug)]
pub struct SequenceGeometryEncoder {
    config: GeometryConfig,
}

impl SequenceGeometryEncoder {
    pub fn new(config: &GeometryConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &GeometryConfig {
        &self.config
    }

    pub fn encode(
        &self,
        _prompt: &GeometryPrompt,
        _image_features: &[Tensor],
        _image_pos_embeds: &[Tensor],
    ) -> Result<EncodedPrompt> {
        candle::bail!("sam3 geometry encoder scaffold only: box/point/mask prompt sequence encoding is not implemented yet")
    }
}
