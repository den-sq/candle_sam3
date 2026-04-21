use super::super::Config;

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

#[derive(Debug, Clone, PartialEq)]
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
    pub suppress_overlapping_based_on_recent_occlusion_threshold: f32,
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
        suppress_overlapping_based_on_recent_occlusion_threshold: 0.7,
        masklet_confirmation_enable: false,
        masklet_confirmation_consecutive_det_thresh: 3,
        compile_all_components: false,
    }
}

pub(crate) fn create_shape_spec(
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
