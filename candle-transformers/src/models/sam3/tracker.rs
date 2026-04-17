use std::collections::BTreeMap;

use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::{checkpoint::Sam3CheckpointSource, neck::VisualBackboneOutput, Config};

const STRICT_PORT_IN_PROGRESS: &str = "SAM3 tracker strict port in progress; legacy tracker implementation was removed. See candle-transformers/src/models/sam3/VIDEO_TRACKER_STRICT_PORT.md before implementing tracker behavior.";

#[derive(Debug, Clone)]
pub struct Sam3TrackerConfig {
    pub image_size: usize,
    pub hidden_dim: usize,
    pub memory_dim: usize,
    pub backbone_stride: usize,
    pub num_maskmem: usize,
    pub max_cond_frames_in_attn: usize,
    pub keep_first_cond_frame: bool,
    pub max_obj_ptrs_in_encoder: usize,
    pub tracker_num_heads: usize,
    pub tracker_num_layers: usize,
    pub tracker_feedforward_dim: usize,
    pub maskmem_interpol_size: usize,
    pub memory_temporal_stride_for_eval: usize,
    pub dynamic_multimask_via_stability: bool,
    pub dynamic_multimask_stability_delta: f32,
    pub dynamic_multimask_stability_thresh: f32,
    pub multimask_output_in_sam: bool,
    pub multimask_output_for_tracking: bool,
    pub multimask_min_pt_num: usize,
    pub multimask_max_pt_num: usize,
}

impl Default for Sam3TrackerConfig {
    fn default() -> Self {
        Self {
            image_size: 1008,
            hidden_dim: 256,
            memory_dim: 64,
            backbone_stride: 14,
            num_maskmem: 7,
            max_cond_frames_in_attn: 4,
            keep_first_cond_frame: false,
            max_obj_ptrs_in_encoder: 16,
            tracker_num_heads: 1,
            tracker_num_layers: 4,
            tracker_feedforward_dim: 2048,
            maskmem_interpol_size: 1152,
            memory_temporal_stride_for_eval: 1,
            dynamic_multimask_via_stability: true,
            dynamic_multimask_stability_delta: 0.05,
            dynamic_multimask_stability_thresh: 0.98,
            multimask_output_in_sam: true,
            multimask_output_for_tracking: true,
            multimask_min_pt_num: 0,
            multimask_max_pt_num: 1,
        }
    }
}

impl Sam3TrackerConfig {
    pub fn from_sam3_config(config: &Config) -> Self {
        Self {
            image_size: config.image.image_size,
            hidden_dim: config.neck.d_model,
            ..Self::default()
        }
    }

    fn image_embedding_size(&self) -> usize {
        self.image_size / self.backbone_stride
    }

    fn low_res_mask_size(&self) -> usize {
        self.image_embedding_size() * 4
    }
}

#[derive(Debug, Clone)]
pub struct TrackerFrameState {
    pub low_res_masks: Tensor,
    pub high_res_masks: Tensor,
    pub iou_scores: Tensor,
    pub obj_ptr: Tensor,
    pub object_score_logits: Tensor,
    pub maskmem_features: Option<Tensor>,
    pub maskmem_pos_enc: Option<Tensor>,
    pub is_cond_frame: bool,
}

impl TrackerFrameState {
    pub fn to_storage_device(&self, device: &candle::Device) -> Result<Self> {
        Ok(Self {
            low_res_masks: self.low_res_masks.to_device(device)?,
            high_res_masks: self.high_res_masks.to_device(device)?,
            iou_scores: self.iou_scores.to_device(device)?,
            obj_ptr: self.obj_ptr.to_device(device)?,
            object_score_logits: self.object_score_logits.to_device(device)?,
            maskmem_features: self
                .maskmem_features
                .as_ref()
                .map(|tensor| tensor.to_device(device))
                .transpose()?,
            maskmem_pos_enc: self
                .maskmem_pos_enc
                .as_ref()
                .map(|tensor| tensor.to_device(device))
                .transpose()?,
            is_cond_frame: self.is_cond_frame,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrackerStepOutput {
    pub state: TrackerFrameState,
    pub prompt_frame_indices: Vec<usize>,
    pub memory_frame_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct Sam3TrackerModel {
    config: Sam3TrackerConfig,
}

impl Sam3TrackerModel {
    pub fn new(config: &Sam3TrackerConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn from_checkpoint_source(
        sam3_config: &Config,
        _checkpoint: &Sam3CheckpointSource,
        _dtype: DType,
        _device: &candle::Device,
    ) -> Result<Self> {
        let tracker_config = Sam3TrackerConfig::from_sam3_config(sam3_config);
        Self::new(
            &tracker_config,
            VarBuilder::zeros(DType::F32, &candle::Device::Cpu),
        )
    }

    pub fn config(&self) -> &Sam3TrackerConfig {
        &self.config
    }

    pub fn input_mask_size(&self) -> usize {
        self.config.low_res_mask_size() * 4
    }

    pub fn track_frame(
        &self,
        _visual: &VisualBackboneOutput,
        _frame_idx: usize,
        _num_frames: usize,
        _point_coords: Option<&Tensor>,
        _point_labels: Option<&Tensor>,
        _boxes_xyxy: Option<&Tensor>,
        _mask_input: Option<&Tensor>,
        _history: &BTreeMap<usize, TrackerFrameState>,
        _is_conditioning_frame: bool,
        _reverse: bool,
        _use_prev_mem_frame: bool,
        _run_mem_encoder: bool,
    ) -> Result<TrackerStepOutput> {
        candle::bail!("{STRICT_PORT_IN_PROGRESS}")
    }

    pub fn encode_state_memory(
        &self,
        _visual: &VisualBackboneOutput,
        _state: &TrackerFrameState,
    ) -> Result<(Tensor, Tensor)> {
        candle::bail!("{STRICT_PORT_IN_PROGRESS}")
    }

    pub fn encode_external_memory(
        &self,
        _visual: &VisualBackboneOutput,
        _high_res_masks: &Tensor,
        _object_score_logits: &Tensor,
        _is_mask_from_points: bool,
    ) -> Result<(Tensor, Tensor)> {
        candle::bail!("{STRICT_PORT_IN_PROGRESS}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::models::sam3::{
        DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig, SegmentationConfig,
        TextConfig, VisionConfig,
    };

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
                d_model: 32,
                width: 64,
                heads: 4,
                layers: 1,
                context_length: 4,
                vocab_size: 64,
            },
            neck: NeckConfig {
                d_model: 32,
                scale_factors: [4.0, 2.0, 1.0, 0.5],
                scalp: 1,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 32,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 64,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 32,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 64,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 32,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 64,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: true,
                hidden_dim: 32,
                upsampling_stages: 1,
                aux_masks: false,
                presence_head: false,
            },
        }
    }

    fn dummy_visual(device: &candle::Device) -> Result<VisualBackboneOutput> {
        let feat = Tensor::zeros((1, 32, 4, 4), DType::F32, device)?;
        let pos = Tensor::zeros((1, 32, 4, 4), DType::F32, device)?;
        Ok(VisualBackboneOutput {
            backbone_fpn: vec![feat.clone()],
            vision_pos_enc: vec![pos.clone()],
            sam2_backbone_fpn: None,
            sam2_pos_enc: None,
        })
    }

    fn dummy_state(device: &candle::Device) -> Result<TrackerFrameState> {
        Ok(TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, device)?,
            high_res_masks: Tensor::zeros((1, 1, 56, 56), DType::F32, device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, device)?,
            obj_ptr: Tensor::zeros((1, 32), DType::F32, device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: true,
        })
    }

    #[test]
    fn tracker_config_defaults_match_upstream_contract() {
        let config = Sam3TrackerConfig::default();
        assert_eq!(config.image_size, 1008);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.memory_dim, 64);
        assert_eq!(config.backbone_stride, 14);
        assert_eq!(config.num_maskmem, 7);
        assert!(config.multimask_output_in_sam);
        assert!(config.multimask_output_for_tracking);
        assert!(config.dynamic_multimask_via_stability);
    }

    #[test]
    fn tracker_config_maps_image_and_hidden_dims_from_sam3_config() {
        let config = Sam3TrackerConfig::from_sam3_config(&tiny_config());
        assert_eq!(config.image_size, 56);
        assert_eq!(config.hidden_dim, 32);
        assert_eq!(config.memory_dim, 64);
    }

    #[test]
    fn tracker_input_mask_size_matches_upstream_contract() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        assert_eq!(model.input_mask_size(), 64);
        Ok(())
    }

    #[test]
    fn tracker_track_frame_reports_strict_port_status() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let err = model
            .track_frame(
                &dummy_visual(&device)?,
                0,
                1,
                None,
                None,
                None,
                None,
                &BTreeMap::new(),
                true,
                false,
                false,
                false,
            )
            .expect_err("strict tracker scaffold should not execute tracking");
        assert!(err.to_string().contains("strict port in progress"));
        Ok(())
    }

    #[test]
    fn tracker_memory_encoding_reports_strict_port_status() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let err = model
            .encode_state_memory(&dummy_visual(&device)?, &dummy_state(&device)?)
            .expect_err("strict tracker scaffold should not encode memory");
        assert!(err.to_string().contains("strict port in progress"));
        Ok(())
    }
}
