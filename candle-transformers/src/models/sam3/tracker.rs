use std::collections::BTreeMap;

use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::{checkpoint::Sam3CheckpointSource, neck::VisualBackboneOutput, Config};

const STRICT_PORT_IN_PROGRESS: &str = "SAM3 tracker strict port in progress; legacy tracker implementation was removed. See candle-transformers/src/models/sam3/VIDEO_TRACKER_STRICT_PORT.md before implementing tracker behavior.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sam3TrackerActivation {
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerPositionEncodingConfig {
    pub num_pos_feats: usize,
    pub normalize: bool,
    pub scale: Option<f32>,
    pub temperature: f32,
    pub precompute_resolution: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3TrackerMaskDownsamplerConfig {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub interpol_size: [usize; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerCxBlockConfig {
    pub dim: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub layer_scale_init_value: f32,
    pub use_dwconv: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3TrackerFuserConfig {
    pub num_layers: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerMaskmemBackboneConfig {
    pub out_dim: usize,
    pub position_encoding: Sam3TrackerPositionEncodingConfig,
    pub mask_downsampler: Sam3TrackerMaskDownsamplerConfig,
    pub cx_block: Sam3TrackerCxBlockConfig,
    pub fuser: Sam3TrackerFuserConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerAttentionConfig {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub downsample_rate: usize,
    pub dropout: f32,
    pub kv_in_dim: Option<usize>,
    pub rope_theta: f32,
    pub feat_sizes: [usize; 2],
    pub rope_k_repeat: bool,
    pub use_fa3: bool,
    pub use_rope_real: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerTransformerLayerConfig {
    pub cross_attention_first: bool,
    pub activation: Sam3TrackerActivation,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub pos_enc_at_attn: bool,
    pub pre_norm: bool,
    pub d_model: usize,
    pub pos_enc_at_cross_attn_keys: bool,
    pub pos_enc_at_cross_attn_queries: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3TrackerTransformerEncoderConfig {
    pub remove_cross_attention_layers: Vec<usize>,
    pub batch_first: bool,
    pub d_model: usize,
    pub frozen: bool,
    pub pos_enc_at_input: bool,
    pub num_layers: usize,
    pub use_act_checkpoint: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerTransformerConfig {
    pub self_attention: Sam3TrackerAttentionConfig,
    pub cross_attention: Sam3TrackerAttentionConfig,
    pub layer: Sam3TrackerTransformerLayerConfig,
    pub encoder: Sam3TrackerTransformerEncoderConfig,
    pub d_model: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3TrackerPromptEncoderConfig {
    pub embed_dim: usize,
    pub image_embedding_size: [usize; 2],
    pub input_image_size: [usize; 2],
    pub mask_in_chans: usize,
    pub mask_input_size: [usize; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerMaskDecoderConfig {
    pub num_multimask_outputs: usize,
    pub transformer_depth: usize,
    pub transformer_embedding_dim: usize,
    pub transformer_mlp_dim: usize,
    pub transformer_num_heads: usize,
    pub transformer_dim: usize,
    pub iou_head_depth: usize,
    pub iou_head_hidden_dim: usize,
    pub use_high_res_features: bool,
    pub iou_prediction_use_sigmoid: bool,
    pub pred_obj_scores: bool,
    pub pred_obj_scores_mlp: bool,
    pub use_multimask_token_for_obj_ptr: bool,
    pub dynamic_multimask_via_stability: bool,
    pub dynamic_multimask_stability_delta: f32,
    pub dynamic_multimask_stability_thresh: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3TrackerPredictorConfig {
    pub with_backbone: bool,
    pub forward_backbone_per_frame_for_eval: bool,
    pub trim_past_non_cond_mem_for_eval: bool,
    pub offload_output_to_cpu_for_eval: bool,
    pub clear_non_cond_mem_around_input: bool,
    pub clear_non_cond_mem_for_multi_obj: bool,
    pub fill_hole_area: usize,
    pub always_start_from_first_ann_frame: bool,
    pub max_point_num_in_prompt_enc: usize,
    pub non_overlap_masks_for_output: bool,
    pub iter_use_prev_mask_pred: bool,
    pub add_all_frames_to_correct_as_cond: bool,
    pub use_prev_mem_frame: bool,
    pub use_stateless_refinement: bool,
    pub refinement_detector_cond_frame_removal_window: usize,
    pub hotstart_delay: usize,
    pub hotstart_unmatch_thresh: usize,
    pub hotstart_dup_thresh: usize,
    pub masklet_confirmation_enable: bool,
    pub masklet_confirmation_consecutive_det_thresh: usize,
    pub compile_all_components: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3TrackerShapeSpec {
    pub image_embedding_size: usize,
    pub low_res_mask_size: usize,
    pub input_mask_size: usize,
    pub attention_feat_sizes: [usize; 2],
    pub mask_downsample_weight_shape: [usize; 4],
    pub maskmem_tpos_enc_shape: [usize; 4],
    pub no_mem_embed_shape: [usize; 3],
    pub no_mem_pos_enc_shape: [usize; 3],
    pub no_obj_ptr_shape: [usize; 2],
    pub no_obj_embed_spatial_shape: [usize; 2],
    pub obj_ptr_proj_weight_shapes: Vec<[usize; 2]>,
    pub obj_ptr_proj_bias_shapes: Vec<[usize; 1]>,
    pub obj_ptr_tpos_proj_weight_shape: [usize; 2],
    pub obj_ptr_tpos_proj_bias_shape: [usize; 1],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sam3TrackerConfig {
    pub image_size: usize,
    pub hidden_dim: usize,
    pub memory_dim: usize,
    pub backbone_stride: usize,
    pub num_maskmem: usize,
    pub max_cond_frames_in_attn: usize,
    pub keep_first_cond_frame: bool,
    pub max_obj_ptrs_in_encoder: usize,
    pub memory_temporal_stride_for_eval: usize,
    pub non_overlap_masks_for_mem_enc: bool,
    pub multimask_output_in_sam: bool,
    pub multimask_output_for_tracking: bool,
    pub multimask_min_pt_num: usize,
    pub multimask_max_pt_num: usize,
    pub use_memory_selection: bool,
    pub mf_threshold: f32,
    pub sigmoid_scale_for_mem_enc: f32,
    pub sigmoid_bias_for_mem_enc: f32,
    pub maskmem_backbone: Sam3TrackerMaskmemBackboneConfig,
    pub transformer: Sam3TrackerTransformerConfig,
    pub prompt_encoder: Sam3TrackerPromptEncoderConfig,
    pub mask_decoder: Sam3TrackerMaskDecoderConfig,
    pub predictor: Sam3TrackerPredictorConfig,
    pub shapes: Sam3TrackerShapeSpec,
}

pub fn create_tracker_maskmem_backbone_config(
    image_size: usize,
    input_mask_size: usize,
) -> Sam3TrackerMaskmemBackboneConfig {
    Sam3TrackerMaskmemBackboneConfig {
        out_dim: 64,
        position_encoding: Sam3TrackerPositionEncodingConfig {
            num_pos_feats: 64,
            normalize: true,
            scale: None,
            temperature: 10_000.0,
            precompute_resolution: image_size,
        },
        mask_downsampler: Sam3TrackerMaskDownsamplerConfig {
            kernel_size: 3,
            stride: 2,
            padding: 1,
            interpol_size: [input_mask_size, input_mask_size],
        },
        cx_block: Sam3TrackerCxBlockConfig {
            dim: 256,
            kernel_size: 7,
            padding: 3,
            layer_scale_init_value: 1.0e-6,
            use_dwconv: true,
        },
        fuser: Sam3TrackerFuserConfig { num_layers: 2 },
    }
}

pub fn create_tracker_transformer_config(
    hidden_dim: usize,
    memory_dim: usize,
    image_embedding_size: usize,
) -> Sam3TrackerTransformerConfig {
    let feat_sizes = [image_embedding_size, image_embedding_size];
    Sam3TrackerTransformerConfig {
        self_attention: Sam3TrackerAttentionConfig {
            embedding_dim: hidden_dim,
            num_heads: 1,
            downsample_rate: 1,
            dropout: 0.1,
            kv_in_dim: None,
            rope_theta: 10_000.0,
            feat_sizes,
            rope_k_repeat: false,
            use_fa3: false,
            use_rope_real: false,
        },
        cross_attention: Sam3TrackerAttentionConfig {
            embedding_dim: hidden_dim,
            num_heads: 1,
            downsample_rate: 1,
            dropout: 0.1,
            kv_in_dim: Some(memory_dim),
            rope_theta: 10_000.0,
            feat_sizes,
            rope_k_repeat: true,
            use_fa3: false,
            use_rope_real: false,
        },
        layer: Sam3TrackerTransformerLayerConfig {
            cross_attention_first: false,
            activation: Sam3TrackerActivation::Relu,
            dim_feedforward: 2048,
            dropout: 0.1,
            pos_enc_at_attn: false,
            pre_norm: true,
            d_model: hidden_dim,
            pos_enc_at_cross_attn_keys: true,
            pos_enc_at_cross_attn_queries: false,
        },
        encoder: Sam3TrackerTransformerEncoderConfig {
            remove_cross_attention_layers: vec![],
            batch_first: true,
            d_model: hidden_dim,
            frozen: false,
            pos_enc_at_input: true,
            num_layers: 4,
            use_act_checkpoint: false,
        },
        d_model: hidden_dim,
    }
}

fn create_prompt_encoder_config(
    hidden_dim: usize,
    image_size: usize,
    image_embedding_size: usize,
    low_res_mask_size: usize,
) -> Sam3TrackerPromptEncoderConfig {
    Sam3TrackerPromptEncoderConfig {
        embed_dim: hidden_dim,
        image_embedding_size: [image_embedding_size, image_embedding_size],
        input_image_size: [image_size, image_size],
        mask_in_chans: 16,
        mask_input_size: [low_res_mask_size, low_res_mask_size],
    }
}

fn create_mask_decoder_config(
    hidden_dim: usize,
    dynamic_multimask_via_stability: bool,
    dynamic_multimask_stability_delta: f32,
    dynamic_multimask_stability_thresh: f32,
) -> Sam3TrackerMaskDecoderConfig {
    Sam3TrackerMaskDecoderConfig {
        num_multimask_outputs: 3,
        transformer_depth: 2,
        transformer_embedding_dim: hidden_dim,
        transformer_mlp_dim: 2048,
        transformer_num_heads: 8,
        transformer_dim: hidden_dim,
        iou_head_depth: 3,
        iou_head_hidden_dim: 256,
        use_high_res_features: true,
        iou_prediction_use_sigmoid: true,
        pred_obj_scores: true,
        pred_obj_scores_mlp: true,
        use_multimask_token_for_obj_ptr: true,
        dynamic_multimask_via_stability,
        dynamic_multimask_stability_delta,
        dynamic_multimask_stability_thresh,
    }
}

fn create_predictor_config(
    with_backbone: bool,
    apply_temporal_disambiguation: bool,
) -> Sam3TrackerPredictorConfig {
    Sam3TrackerPredictorConfig {
        with_backbone,
        forward_backbone_per_frame_for_eval: true,
        trim_past_non_cond_mem_for_eval: false,
        offload_output_to_cpu_for_eval: false,
        clear_non_cond_mem_around_input: true,
        clear_non_cond_mem_for_multi_obj: false,
        fill_hole_area: 16,
        always_start_from_first_ann_frame: false,
        max_point_num_in_prompt_enc: 16,
        non_overlap_masks_for_output: false,
        iter_use_prev_mask_pred: true,
        add_all_frames_to_correct_as_cond: true,
        use_prev_mem_frame: false,
        use_stateless_refinement: false,
        refinement_detector_cond_frame_removal_window: 16,
        hotstart_delay: if apply_temporal_disambiguation { 15 } else { 0 },
        hotstart_unmatch_thresh: if apply_temporal_disambiguation { 8 } else { 0 },
        hotstart_dup_thresh: if apply_temporal_disambiguation { 8 } else { 0 },
        masklet_confirmation_enable: false,
        masklet_confirmation_consecutive_det_thresh: 3,
        compile_all_components: false,
    }
}

fn create_shape_spec(
    image_size: usize,
    hidden_dim: usize,
    memory_dim: usize,
    backbone_stride: usize,
    num_maskmem: usize,
) -> Sam3TrackerShapeSpec {
    let image_embedding_size = image_size / backbone_stride;
    let low_res_mask_size = image_embedding_size * 4;
    let input_mask_size = low_res_mask_size * 4;
    Sam3TrackerShapeSpec {
        image_embedding_size,
        low_res_mask_size,
        input_mask_size,
        attention_feat_sizes: [image_embedding_size, image_embedding_size],
        mask_downsample_weight_shape: [1, 1, 4, 4],
        maskmem_tpos_enc_shape: [num_maskmem, 1, 1, memory_dim],
        no_mem_embed_shape: [1, 1, hidden_dim],
        no_mem_pos_enc_shape: [1, 1, hidden_dim],
        no_obj_ptr_shape: [1, hidden_dim],
        no_obj_embed_spatial_shape: [1, memory_dim],
        obj_ptr_proj_weight_shapes: vec![[hidden_dim, hidden_dim]; 3],
        obj_ptr_proj_bias_shapes: vec![[hidden_dim]; 3],
        obj_ptr_tpos_proj_weight_shape: [memory_dim, hidden_dim],
        obj_ptr_tpos_proj_bias_shape: [memory_dim],
    }
}

impl Default for Sam3TrackerConfig {
    fn default() -> Self {
        Self::build_tracker(false)
    }
}

impl Sam3TrackerConfig {
    pub fn build_tracker(apply_temporal_disambiguation: bool) -> Self {
        Self::from_dimensions(1008, 256, 14, false, apply_temporal_disambiguation)
    }

    pub fn from_sam3_config(config: &Config) -> Self {
        Self::from_dimensions(
            config.image.image_size,
            config.neck.d_model,
            config.vision.patch_size,
            false,
            false,
        )
    }

    fn from_dimensions(
        image_size: usize,
        hidden_dim: usize,
        backbone_stride: usize,
        with_backbone: bool,
        apply_temporal_disambiguation: bool,
    ) -> Self {
        let memory_dim = 64;
        let num_maskmem = 7;
        let dynamic_multimask_via_stability = true;
        let dynamic_multimask_stability_delta = 0.05;
        let dynamic_multimask_stability_thresh = 0.98;
        let shapes = create_shape_spec(
            image_size,
            hidden_dim,
            memory_dim,
            backbone_stride,
            num_maskmem,
        );
        Self {
            image_size,
            hidden_dim,
            memory_dim,
            backbone_stride,
            num_maskmem,
            max_cond_frames_in_attn: 4,
            keep_first_cond_frame: false,
            max_obj_ptrs_in_encoder: 16,
            memory_temporal_stride_for_eval: 1,
            non_overlap_masks_for_mem_enc: false,
            multimask_output_in_sam: true,
            multimask_output_for_tracking: true,
            multimask_min_pt_num: 0,
            multimask_max_pt_num: 1,
            use_memory_selection: apply_temporal_disambiguation,
            mf_threshold: 0.01,
            sigmoid_scale_for_mem_enc: 20.0,
            sigmoid_bias_for_mem_enc: -10.0,
            maskmem_backbone: create_tracker_maskmem_backbone_config(
                image_size,
                shapes.input_mask_size,
            ),
            transformer: create_tracker_transformer_config(
                hidden_dim,
                memory_dim,
                shapes.image_embedding_size,
            ),
            prompt_encoder: create_prompt_encoder_config(
                hidden_dim,
                image_size,
                shapes.image_embedding_size,
                shapes.low_res_mask_size,
            ),
            mask_decoder: create_mask_decoder_config(
                hidden_dim,
                dynamic_multimask_via_stability,
                dynamic_multimask_stability_delta,
                dynamic_multimask_stability_thresh,
            ),
            predictor: create_predictor_config(with_backbone, apply_temporal_disambiguation),
            shapes,
        }
    }

    pub fn image_embedding_size(&self) -> usize {
        self.shapes.image_embedding_size
    }

    pub fn low_res_mask_size(&self) -> usize {
        self.shapes.low_res_mask_size
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

    pub fn image_embedding_size(&self) -> usize {
        self.config.image_embedding_size()
    }

    pub fn low_res_mask_size(&self) -> usize {
        self.config.low_res_mask_size()
    }

    pub fn input_mask_size(&self) -> usize {
        self.config.shapes.input_mask_size
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

    use std::{collections::HashMap, fs, path::PathBuf};

    use serde::Deserialize;

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

    fn expected_upstream_config(apply_temporal_disambiguation: bool) -> Sam3TrackerConfig {
        Sam3TrackerConfig {
            image_size: 1008,
            hidden_dim: 256,
            memory_dim: 64,
            backbone_stride: 14,
            num_maskmem: 7,
            max_cond_frames_in_attn: 4,
            keep_first_cond_frame: false,
            max_obj_ptrs_in_encoder: 16,
            memory_temporal_stride_for_eval: 1,
            non_overlap_masks_for_mem_enc: false,
            multimask_output_in_sam: true,
            multimask_output_for_tracking: true,
            multimask_min_pt_num: 0,
            multimask_max_pt_num: 1,
            use_memory_selection: apply_temporal_disambiguation,
            mf_threshold: 0.01,
            sigmoid_scale_for_mem_enc: 20.0,
            sigmoid_bias_for_mem_enc: -10.0,
            maskmem_backbone: Sam3TrackerMaskmemBackboneConfig {
                out_dim: 64,
                position_encoding: Sam3TrackerPositionEncodingConfig {
                    num_pos_feats: 64,
                    normalize: true,
                    scale: None,
                    temperature: 10_000.0,
                    precompute_resolution: 1008,
                },
                mask_downsampler: Sam3TrackerMaskDownsamplerConfig {
                    kernel_size: 3,
                    stride: 2,
                    padding: 1,
                    interpol_size: [1152, 1152],
                },
                cx_block: Sam3TrackerCxBlockConfig {
                    dim: 256,
                    kernel_size: 7,
                    padding: 3,
                    layer_scale_init_value: 1.0e-6,
                    use_dwconv: true,
                },
                fuser: Sam3TrackerFuserConfig { num_layers: 2 },
            },
            transformer: Sam3TrackerTransformerConfig {
                self_attention: Sam3TrackerAttentionConfig {
                    embedding_dim: 256,
                    num_heads: 1,
                    downsample_rate: 1,
                    dropout: 0.1,
                    kv_in_dim: None,
                    rope_theta: 10_000.0,
                    feat_sizes: [72, 72],
                    rope_k_repeat: false,
                    use_fa3: false,
                    use_rope_real: false,
                },
                cross_attention: Sam3TrackerAttentionConfig {
                    embedding_dim: 256,
                    num_heads: 1,
                    downsample_rate: 1,
                    dropout: 0.1,
                    kv_in_dim: Some(64),
                    rope_theta: 10_000.0,
                    feat_sizes: [72, 72],
                    rope_k_repeat: true,
                    use_fa3: false,
                    use_rope_real: false,
                },
                layer: Sam3TrackerTransformerLayerConfig {
                    cross_attention_first: false,
                    activation: Sam3TrackerActivation::Relu,
                    dim_feedforward: 2048,
                    dropout: 0.1,
                    pos_enc_at_attn: false,
                    pre_norm: true,
                    d_model: 256,
                    pos_enc_at_cross_attn_keys: true,
                    pos_enc_at_cross_attn_queries: false,
                },
                encoder: Sam3TrackerTransformerEncoderConfig {
                    remove_cross_attention_layers: vec![],
                    batch_first: true,
                    d_model: 256,
                    frozen: false,
                    pos_enc_at_input: true,
                    num_layers: 4,
                    use_act_checkpoint: false,
                },
                d_model: 256,
            },
            prompt_encoder: Sam3TrackerPromptEncoderConfig {
                embed_dim: 256,
                image_embedding_size: [72, 72],
                input_image_size: [1008, 1008],
                mask_in_chans: 16,
                mask_input_size: [288, 288],
            },
            mask_decoder: Sam3TrackerMaskDecoderConfig {
                num_multimask_outputs: 3,
                transformer_depth: 2,
                transformer_embedding_dim: 256,
                transformer_mlp_dim: 2048,
                transformer_num_heads: 8,
                transformer_dim: 256,
                iou_head_depth: 3,
                iou_head_hidden_dim: 256,
                use_high_res_features: true,
                iou_prediction_use_sigmoid: true,
                pred_obj_scores: true,
                pred_obj_scores_mlp: true,
                use_multimask_token_for_obj_ptr: true,
                dynamic_multimask_via_stability: true,
                dynamic_multimask_stability_delta: 0.05,
                dynamic_multimask_stability_thresh: 0.98,
            },
            predictor: Sam3TrackerPredictorConfig {
                with_backbone: false,
                forward_backbone_per_frame_for_eval: true,
                trim_past_non_cond_mem_for_eval: false,
                offload_output_to_cpu_for_eval: false,
                clear_non_cond_mem_around_input: true,
                clear_non_cond_mem_for_multi_obj: false,
                fill_hole_area: 16,
                always_start_from_first_ann_frame: false,
                max_point_num_in_prompt_enc: 16,
                non_overlap_masks_for_output: false,
                iter_use_prev_mask_pred: true,
                add_all_frames_to_correct_as_cond: true,
                use_prev_mem_frame: false,
                use_stateless_refinement: false,
                refinement_detector_cond_frame_removal_window: 16,
                hotstart_delay: if apply_temporal_disambiguation { 15 } else { 0 },
                hotstart_unmatch_thresh: if apply_temporal_disambiguation { 8 } else { 0 },
                hotstart_dup_thresh: if apply_temporal_disambiguation { 8 } else { 0 },
                masklet_confirmation_enable: false,
                masklet_confirmation_consecutive_det_thresh: 3,
                compile_all_components: false,
            },
            shapes: Sam3TrackerShapeSpec {
                image_embedding_size: 72,
                low_res_mask_size: 288,
                input_mask_size: 1152,
                attention_feat_sizes: [72, 72],
                mask_downsample_weight_shape: [1, 1, 4, 4],
                maskmem_tpos_enc_shape: [7, 1, 1, 64],
                no_mem_embed_shape: [1, 1, 256],
                no_mem_pos_enc_shape: [1, 1, 256],
                no_obj_ptr_shape: [1, 256],
                no_obj_embed_spatial_shape: [1, 64],
                obj_ptr_proj_weight_shapes: vec![[256, 256], [256, 256], [256, 256]],
                obj_ptr_proj_bias_shapes: vec![[256], [256], [256]],
                obj_ptr_tpos_proj_weight_shape: [64, 256],
                obj_ptr_tpos_proj_bias_shape: [64],
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

    #[derive(Debug, Deserialize)]
    struct TrackerInternalManifest {
        tracker_config: TrackerFixtureConfig,
        predictor_config: TrackerPredictorFixtureConfig,
        records: Vec<TrackerInternalRecord>,
    }

    #[derive(Debug, Deserialize)]
    struct TrackerMaskDecoderExtraArgsFixtureConfig {
        dynamic_multimask_via_stability: bool,
        dynamic_multimask_stability_delta: f32,
        dynamic_multimask_stability_thresh: f32,
    }

    #[derive(Debug, Deserialize)]
    struct TrackerFixtureConfig {
        with_backbone: bool,
        image_size: usize,
        backbone_stride: usize,
        low_res_mask_size: usize,
        input_mask_size: usize,
        num_maskmem: usize,
        max_cond_frames_in_attn: usize,
        keep_first_cond_frame: bool,
        memory_temporal_stride_for_eval: usize,
        max_obj_ptrs_in_encoder: usize,
        non_overlap_masks_for_mem_enc: bool,
        forward_backbone_per_frame_for_eval: bool,
        trim_past_non_cond_mem_for_eval: bool,
        offload_output_to_cpu_for_eval: bool,
        sigmoid_scale_for_mem_enc: f32,
        sigmoid_bias_for_mem_enc: f32,
        multimask_output_in_sam: bool,
        multimask_output_for_tracking: bool,
        multimask_min_pt_num: usize,
        multimask_max_pt_num: usize,
        use_memory_selection: bool,
        mf_threshold: f32,
        input_mask_binarize_threshold: f32,
        video_mask_binarize_threshold: f32,
        mask_as_output_out_scale: f32,
        mask_as_output_out_bias: f32,
        memory_prompt_mask_threshold: f32,
        sam_mask_decoder_extra_args: TrackerMaskDecoderExtraArgsFixtureConfig,
    }

    #[derive(Debug, Deserialize)]
    struct TrackerPredictorFixtureConfig {
        compile_model: bool,
        clear_non_cond_mem_around_input: bool,
        clear_non_cond_mem_for_multi_obj: bool,
        fill_hole_area: usize,
        hotstart_delay: usize,
        hotstart_unmatch_thresh: usize,
        hotstart_dup_thresh: usize,
        masklet_confirmation_enable: bool,
        masklet_confirmation_consecutive_det_thresh: usize,
        always_start_from_first_ann_frame: bool,
        max_point_num_in_prompt_enc: usize,
        non_overlap_masks_for_output: bool,
        iter_use_prev_mask_pred: bool,
        add_all_frames_to_correct_as_cond: bool,
        use_prev_mem_frame: bool,
        use_stateless_refinement: bool,
        refinement_detector_cond_frame_removal_window: usize,
    }

    #[derive(Debug, Deserialize)]
    struct TrackerInternalRecord {
        stage: String,
        frame_idx: usize,
        metadata: serde_json::Value,
        tensor_keys: HashMap<String, String>,
        tensor_stats: HashMap<String, TrackerTensorStat>,
    }

    #[derive(Debug, Deserialize)]
    struct TrackerTensorStat {
        shape: Vec<usize>,
        dtype: String,
    }

    #[derive(Debug, Clone, Copy)]
    enum TrackerFixtureBundle {
        Default,
        TemporalDisambiguation,
    }

    impl TrackerFixtureBundle {
        fn debug_dir(self) -> &'static str {
            match self {
                Self::Default => "../candle-examples/examples/sam3/reference_video_box_debug/debug",
                Self::TemporalDisambiguation => {
                    "../candle-examples/examples/sam3/reference_video_box_debug_temporal_disambiguation/debug"
                }
            }
        }
    }

    fn tracker_fixture_dir(bundle: TrackerFixtureBundle) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(bundle.debug_dir())
    }

    fn load_tracker_internal_manifest(bundle: TrackerFixtureBundle) -> Result<TrackerInternalManifest> {
        let path = tracker_fixture_dir(bundle).join("internal_manifest.json");
        let contents = fs::read_to_string(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to read tracker internal manifest {}: {err}",
                path.display()
            ))
        })?;
        serde_json::from_str(&contents).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to parse tracker internal manifest {}: {err}",
                path.display()
            ))
        })
    }

    fn tracker_record<'a>(
        manifest: &'a TrackerInternalManifest,
        frame_idx: usize,
        stage: &str,
    ) -> Result<&'a TrackerInternalRecord> {
        manifest
            .records
            .iter()
            .find(|record| record.frame_idx == frame_idx && record.stage == stage)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "missing tracker internal record for frame {frame_idx} stage `{stage}`"
                ))
            })
    }

    fn fixture_shape(record: &TrackerInternalRecord, key: &str) -> Result<Vec<usize>> {
        record
            .tensor_stats
            .get(key)
            .map(|stats| stats.shape.clone())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "tracker internal record frame {} stage `{}` missing tensor stat `{key}`",
                    record.frame_idx, record.stage
                ))
            })
    }

    fn fixture_dtype<'a>(record: &'a TrackerInternalRecord, key: &str) -> Result<&'a str> {
        record
            .tensor_stats
            .get(key)
            .map(|stats| stats.dtype.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "tracker internal record frame {} stage `{}` missing tensor stat `{key}`",
                    record.frame_idx, record.stage
                ))
            })
    }

    #[test]
    fn tracker_build_config_matches_upstream_contract_without_temporal_disambiguation() {
        assert_eq!(
            Sam3TrackerConfig::build_tracker(false),
            expected_upstream_config(false)
        );
    }

    #[test]
    fn tracker_build_config_matches_upstream_contract_with_temporal_disambiguation() {
        assert_eq!(
            Sam3TrackerConfig::build_tracker(true),
            expected_upstream_config(true)
        );
    }

    #[test]
    fn tracker_transformer_contract_matches_upstream_builder() {
        assert_eq!(
            create_tracker_transformer_config(256, 64, 72),
            expected_upstream_config(false).transformer
        );
    }

    #[test]
    fn tracker_maskmem_backbone_contract_matches_upstream_builder() {
        assert_eq!(
            create_tracker_maskmem_backbone_config(1008, 1152),
            expected_upstream_config(false).maskmem_backbone
        );
    }

    #[test]
    fn tracker_shape_spec_matches_constructed_upstream_tensor_shapes() {
        assert_eq!(
            create_shape_spec(1008, 256, 64, 14, 7),
            expected_upstream_config(false).shapes
        );
    }

    #[test]
    fn tracker_config_from_sam3_config_updates_derived_shapes_consistently() {
        let config = Sam3TrackerConfig::from_sam3_config(&tiny_config());
        assert_eq!(config.image_size, 56);
        assert_eq!(config.hidden_dim, 32);
        assert_eq!(config.memory_dim, 64);
        assert_eq!(config.backbone_stride, 14);
        assert_eq!(config.shapes.image_embedding_size, 4);
        assert_eq!(config.shapes.low_res_mask_size, 16);
        assert_eq!(config.shapes.input_mask_size, 64);
        assert_eq!(config.transformer.self_attention.feat_sizes, [4, 4]);
        assert_eq!(config.transformer.cross_attention.feat_sizes, [4, 4]);
        assert_eq!(config.prompt_encoder.image_embedding_size, [4, 4]);
        assert_eq!(config.prompt_encoder.input_image_size, [56, 56]);
        assert_eq!(config.prompt_encoder.mask_input_size, [16, 16]);
        assert_eq!(
            config.maskmem_backbone.mask_downsampler.interpol_size,
            [64, 64]
        );
        assert_eq!(
            config.shapes.obj_ptr_proj_weight_shapes,
            vec![[32, 32], [32, 32], [32, 32]]
        );
        assert_eq!(config.shapes.obj_ptr_tpos_proj_weight_shape, [64, 32]);
    }

    #[test]
    fn tracker_model_exposes_exact_builder_shapes() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::build_tracker(false),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        assert_eq!(model.image_embedding_size(), 72);
        assert_eq!(model.low_res_mask_size(), 288);
        assert_eq!(model.input_mask_size(), 1152);
        Ok(())
    }

    fn assert_fixture_backed_tracker_config_matches_runtime_upstream_bundle(
        bundle: TrackerFixtureBundle,
        apply_temporal_disambiguation: bool,
    ) -> Result<()> {
        let manifest = load_tracker_internal_manifest(bundle)?;
        let fixture = manifest.tracker_config;
        let predictor_fixture = manifest.predictor_config;
        let config = Sam3TrackerConfig::build_tracker(apply_temporal_disambiguation);
        assert_eq!(config.predictor.with_backbone, fixture.with_backbone);
        assert_eq!(config.image_size, fixture.image_size);
        assert_eq!(config.backbone_stride, fixture.backbone_stride);
        assert_eq!(config.low_res_mask_size(), fixture.low_res_mask_size);
        assert_eq!(config.shapes.input_mask_size, fixture.input_mask_size);
        assert_eq!(config.num_maskmem, fixture.num_maskmem);
        assert_eq!(
            config.max_cond_frames_in_attn,
            fixture.max_cond_frames_in_attn
        );
        assert_eq!(config.keep_first_cond_frame, fixture.keep_first_cond_frame);
        assert_eq!(
            config.memory_temporal_stride_for_eval,
            fixture.memory_temporal_stride_for_eval
        );
        assert_eq!(
            config.max_obj_ptrs_in_encoder,
            fixture.max_obj_ptrs_in_encoder
        );
        assert_eq!(
            config.non_overlap_masks_for_mem_enc,
            fixture.non_overlap_masks_for_mem_enc
        );
        assert_eq!(
            config.sigmoid_scale_for_mem_enc,
            fixture.sigmoid_scale_for_mem_enc
        );
        assert_eq!(
            config.sigmoid_bias_for_mem_enc,
            fixture.sigmoid_bias_for_mem_enc
        );
        assert_eq!(
            config.multimask_output_in_sam,
            fixture.multimask_output_in_sam
        );
        assert_eq!(
            config.multimask_output_for_tracking,
            fixture.multimask_output_for_tracking
        );
        assert_eq!(config.multimask_min_pt_num, fixture.multimask_min_pt_num);
        assert_eq!(config.multimask_max_pt_num, fixture.multimask_max_pt_num);
        assert_eq!(config.use_memory_selection, fixture.use_memory_selection);
        assert_eq!(config.mf_threshold, fixture.mf_threshold);
        assert_eq!(
            config.predictor.forward_backbone_per_frame_for_eval,
            fixture.forward_backbone_per_frame_for_eval
        );
        assert_eq!(
            config.predictor.trim_past_non_cond_mem_for_eval,
            fixture.trim_past_non_cond_mem_for_eval
        );
        assert_eq!(
            config.predictor.offload_output_to_cpu_for_eval,
            fixture.offload_output_to_cpu_for_eval
        );
        assert_eq!(
            config.mask_decoder.dynamic_multimask_via_stability,
            fixture
                .sam_mask_decoder_extra_args
                .dynamic_multimask_via_stability
        );
        assert_eq!(
            config.mask_decoder.dynamic_multimask_stability_delta,
            fixture
                .sam_mask_decoder_extra_args
                .dynamic_multimask_stability_delta
        );
        assert_eq!(
            config.mask_decoder.dynamic_multimask_stability_thresh,
            fixture
                .sam_mask_decoder_extra_args
                .dynamic_multimask_stability_thresh
        );
        assert_eq!(fixture.input_mask_binarize_threshold, 0.0);
        assert_eq!(fixture.video_mask_binarize_threshold, 0.5);
        assert_eq!(fixture.mask_as_output_out_scale, 20.0);
        assert_eq!(fixture.mask_as_output_out_bias, -10.0);
        assert_eq!(fixture.memory_prompt_mask_threshold, 0.0);
        assert_eq!(
            config.predictor.fill_hole_area,
            predictor_fixture.fill_hole_area
        );
        assert_eq!(
            config.predictor.clear_non_cond_mem_around_input,
            predictor_fixture.clear_non_cond_mem_around_input
        );
        assert_eq!(
            config.predictor.clear_non_cond_mem_for_multi_obj,
            predictor_fixture.clear_non_cond_mem_for_multi_obj
        );
        assert_eq!(
            config.predictor.always_start_from_first_ann_frame,
            predictor_fixture.always_start_from_first_ann_frame
        );
        assert_eq!(
            config.predictor.max_point_num_in_prompt_enc,
            predictor_fixture.max_point_num_in_prompt_enc
        );
        assert_eq!(
            config.predictor.non_overlap_masks_for_output,
            predictor_fixture.non_overlap_masks_for_output
        );
        assert_eq!(
            config.predictor.iter_use_prev_mask_pred,
            predictor_fixture.iter_use_prev_mask_pred
        );
        assert_eq!(
            config.predictor.add_all_frames_to_correct_as_cond,
            predictor_fixture.add_all_frames_to_correct_as_cond
        );
        assert_eq!(
            config.predictor.use_prev_mem_frame,
            predictor_fixture.use_prev_mem_frame
        );
        assert_eq!(
            config.predictor.use_stateless_refinement,
            predictor_fixture.use_stateless_refinement
        );
        assert_eq!(
            config.predictor.refinement_detector_cond_frame_removal_window,
            predictor_fixture.refinement_detector_cond_frame_removal_window
        );
        assert_eq!(
            config.predictor.hotstart_delay,
            predictor_fixture.hotstart_delay
        );
        assert_eq!(
            config.predictor.hotstart_unmatch_thresh,
            predictor_fixture.hotstart_unmatch_thresh
        );
        assert_eq!(
            config.predictor.hotstart_dup_thresh,
            predictor_fixture.hotstart_dup_thresh
        );
        assert_eq!(
            config.predictor.masklet_confirmation_enable,
            predictor_fixture.masklet_confirmation_enable
        );
        assert_eq!(
            config.predictor.masklet_confirmation_consecutive_det_thresh,
            predictor_fixture.masklet_confirmation_consecutive_det_thresh
        );
        assert_eq!(
            config.predictor.compile_all_components,
            predictor_fixture.compile_model
        );
        Ok(())
    }

    #[test]
    fn fixture_backed_tracker_config_matches_default_runtime_upstream_bundle() -> Result<()> {
        assert_fixture_backed_tracker_config_matches_runtime_upstream_bundle(
            TrackerFixtureBundle::Default,
            false,
        )
    }

    #[test]
    fn fixture_backed_tracker_config_matches_temporal_disambiguation_runtime_upstream_bundle(
    ) -> Result<()> {
        assert_fixture_backed_tracker_config_matches_runtime_upstream_bundle(
            TrackerFixtureBundle::TemporalDisambiguation,
            true,
        )
    }

    fn assert_fixture_backed_tracker_tensor_shapes_match_upstream_runtime_bundle(
        bundle: TrackerFixtureBundle,
        apply_temporal_disambiguation: bool,
    ) -> Result<()> {
        let manifest = load_tracker_internal_manifest(bundle)?;
        let config = Sam3TrackerConfig::build_tracker(apply_temporal_disambiguation);

        let add_new_objects = tracker_record(&manifest, 0, "tracker_add_new_objects_input")?;
        assert_eq!(
            fixture_shape(add_new_objects, "new_object_masks_before_resize")?,
            vec![1, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_dtype(add_new_objects, "new_object_masks_before_resize")?,
            "torch.bfloat16"
        );

        let frame0_track_step = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(
            fixture_shape(frame0_track_step, "current_vision_feats")?,
            vec![
                config.image_embedding_size() * config.image_embedding_size(),
                1,
                config.hidden_dim
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "current_vision_pos_embeds")?,
            vec![
                config.image_embedding_size() * config.image_embedding_size(),
                1,
                config.hidden_dim
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "mask_inputs")?,
            vec![
                1,
                1,
                config.shapes.input_mask_size,
                config.shapes.input_mask_size
            ]
        );
        let expected_mask_input_low_res =
            (config.shapes.input_mask_size / config.backbone_stride) * 4;
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.pred_masks")?,
            vec![
                1,
                1,
                expected_mask_input_low_res,
                expected_mask_input_low_res
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.pred_masks_high_res")?,
            vec![
                1,
                1,
                config.shapes.input_mask_size,
                config.shapes.input_mask_size
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.obj_ptr")?,
            vec![1, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.object_score_logits")?,
            vec![1, 1]
        );

        let frame0_preflight =
            tracker_record(&manifest, 0, "tracker_add_new_objects_post_preflight")?;
        assert_eq!(
            fixture_shape(frame0_preflight, "post_preflight_cond_output.pred_masks")?,
            vec![1, 1, config.shapes.low_res_mask_size, config.shapes.low_res_mask_size]
        );
        assert_eq!(
            fixture_shape(frame0_preflight, "post_preflight_cond_output.obj_ptr")?,
            vec![1, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(
                frame0_preflight,
                "post_preflight_cond_output.object_score_logits"
            )?,
            vec![1, 1]
        );
        assert_eq!(
            fixture_shape(
                frame0_preflight,
                "post_preflight_cond_output.maskmem_features"
            )?,
            vec![
                1,
                config.memory_dim,
                config.image_embedding_size(),
                config.image_embedding_size()
            ]
        );
        assert_eq!(
            fixture_shape(
                frame0_preflight,
                "post_preflight_cond_output.maskmem_pos_enc.0"
            )?,
            vec![
                1,
                config.memory_dim,
                config.image_embedding_size(),
                config.image_embedding_size()
            ]
        );

        for frame_idx in 0..=3 {
            let encode_new_memory = tracker_record(&manifest, frame_idx, "encode_new_memory")?;
            assert_eq!(
                fixture_shape(encode_new_memory, "maskmem_features")?,
                vec![
                    1,
                    config.memory_dim,
                    config.image_embedding_size(),
                    config.image_embedding_size()
                ]
            );
            assert_eq!(
                fixture_shape(encode_new_memory, "maskmem_pos_enc.0")?,
                vec![
                    1,
                    config.memory_dim,
                    config.image_embedding_size(),
                    config.image_embedding_size()
                ]
            );
            assert_eq!(
                fixture_shape(encode_new_memory, "object_score_logits")?,
                vec![1, 1]
            );
        }

        for frame_idx in 1..=3 {
            let prep = tracker_record(&manifest, frame_idx, "prepare_memory_conditioned_features")?;
            assert_eq!(
                fixture_shape(prep, "pix_feat_with_mem")?,
                vec![
                    1,
                    config.hidden_dim,
                    config.image_embedding_size(),
                    config.image_embedding_size()
                ]
            );
            let pointer_frames = prep.metadata["selected_object_pointer_frame_indices"]
                .as_array()
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "frame {frame_idx} prepare_memory_conditioned_features missing selected_object_pointer_frame_indices"
                    ))
                })?;
            assert_eq!(
                fixture_shape(prep, "object_pointer_temporal_pos_enc")?,
                vec![pointer_frames.len(), config.memory_dim]
            );

            let track_step = tracker_record(&manifest, frame_idx, "track_step")?;
            assert_eq!(
                fixture_shape(track_step, "current_vision_feats")?,
                vec![
                    config.image_embedding_size() * config.image_embedding_size(),
                    1,
                    config.hidden_dim
                ]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.pred_masks")?,
                vec![1, 1, config.low_res_mask_size(), config.low_res_mask_size()]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.pred_masks_high_res")?,
                vec![1, 1, config.image_size, config.image_size]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.obj_ptr")?,
                vec![1, config.hidden_dim]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.object_score_logits")?,
                vec![1, 1]
            );
        }

        Ok(())
    }

    #[test]
    fn fixture_backed_tracker_tensor_shapes_match_default_runtime_upstream_bundle() -> Result<()>
    {
        assert_fixture_backed_tracker_tensor_shapes_match_upstream_runtime_bundle(
            TrackerFixtureBundle::Default,
            false,
        )
    }

    #[test]
    fn fixture_backed_tracker_tensor_shapes_match_temporal_disambiguation_runtime_upstream_bundle(
    ) -> Result<()> {
        assert_fixture_backed_tracker_tensor_shapes_match_upstream_runtime_bundle(
            TrackerFixtureBundle::TemporalDisambiguation,
            true,
        )
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
