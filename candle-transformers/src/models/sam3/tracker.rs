use std::collections::BTreeMap;

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Embedding, Module, VarBuilder,
};

use crate::models::segment_anything::{
    linear, prompt_encoder::PromptEncoder, transformer::TwoWayTransformer, LayerNorm2d, Linear,
};

use super::{
    checkpoint::Sam3CheckpointSource,
    neck::{Sam3DualViTDetNeck, VisualBackboneOutput},
    vitdet::Sam3ViTDetTrunk,
    Config,
};

const STRICT_PORT_IN_PROGRESS: &str = "SAM3 tracker strict port in progress; legacy tracker implementation was removed. See candle-transformers/src/models/sam3/VIDEO_TRACKER_STRICT_PORT.md before implementing tracker behavior.";
const NO_OBJ_SCORE: f64 = -1024.0;

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
struct TrackerMlp {
    layers: Vec<Linear>,
    sigmoid_output: bool,
}

impl TrackerMlp {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let vb = vb.pp("layers");
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(linear(vb.pp(i), in_dim, out_dim, true)?);
        }
        Ok(Self {
            layers,
            sigmoid_output,
        })
    }
}

impl Module for TrackerMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if index + 1 < self.layers.len() {
                xs = xs.relu()?;
            }
        }
        if self.sigmoid_output {
            candle_nn::ops::sigmoid(&xs)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug)]
enum PredObjScoreHead {
    Linear(Linear),
    Mlp(TrackerMlp),
}

impl PredObjScoreHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(layer) => layer.forward(xs),
            Self::Mlp(layer) => layer.forward(xs),
        }
    }
}

#[derive(Debug)]
struct Sam3TrackerMaskDecoder {
    transformer_dim: usize,
    transformer: TwoWayTransformer,
    iou_token: Embedding,
    mask_tokens: Embedding,
    obj_score_token: Option<Embedding>,
    output_upscaling_conv1: ConvTranspose2d,
    output_upscaling_ln: LayerNorm2d,
    output_upscaling_conv2: ConvTranspose2d,
    conv_s0: Option<Conv2d>,
    conv_s1: Option<Conv2d>,
    output_hypernetworks_mlps: Vec<TrackerMlp>,
    iou_prediction_head: TrackerMlp,
    pred_obj_score_head: Option<PredObjScoreHead>,
    num_mask_tokens: usize,
    use_high_res_features: bool,
    use_multimask_token_for_obj_ptr: bool,
    dynamic_multimask_via_stability: bool,
    dynamic_multimask_stability_delta: f32,
    dynamic_multimask_stability_thresh: f32,
}

impl Sam3TrackerMaskDecoder {
    fn new(config: &Sam3TrackerMaskDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let num_mask_tokens = config.num_multimask_outputs + 1;
        let transformer = TwoWayTransformer::new(
            config.transformer_depth,
            config.transformer_embedding_dim,
            config.transformer_num_heads,
            config.transformer_mlp_dim,
            vb.pp("transformer"),
        )?;
        let iou_token = candle_nn::embedding(1, config.transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens = candle_nn::embedding(
            num_mask_tokens,
            config.transformer_dim,
            vb.pp("mask_tokens"),
        )?;
        let obj_score_token = if config.pred_obj_scores {
            Some(candle_nn::embedding(
                1,
                config.transformer_dim,
                vb.pp("obj_score_token"),
            )?)
        } else {
            None
        };
        let deconv_cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        let output_upscaling_conv1 = candle_nn::conv_transpose2d(
            config.transformer_dim,
            config.transformer_dim / 4,
            2,
            deconv_cfg,
            vb.pp("output_upscaling.0"),
        )?;
        let output_upscaling_ln = LayerNorm2d::new(
            config.transformer_dim / 4,
            1e-6,
            vb.pp("output_upscaling.1"),
        )?;
        let output_upscaling_conv2 = candle_nn::conv_transpose2d(
            config.transformer_dim / 4,
            config.transformer_dim / 8,
            2,
            deconv_cfg,
            vb.pp("output_upscaling.3"),
        )?;
        let (conv_s0, conv_s1) = if config.use_high_res_features {
            (
                Some(candle_nn::conv2d(
                    config.transformer_dim,
                    config.transformer_dim / 8,
                    1,
                    Default::default(),
                    vb.pp("conv_s0"),
                )?),
                Some(candle_nn::conv2d(
                    config.transformer_dim,
                    config.transformer_dim / 4,
                    1,
                    Default::default(),
                    vb.pp("conv_s1"),
                )?),
            )
        } else {
            (None, None)
        };
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        let output_hypernetworks_vb = vb.pp("output_hypernetworks_mlps");
        for index in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(TrackerMlp::new(
                config.transformer_dim,
                config.transformer_dim,
                config.transformer_dim / 8,
                3,
                false,
                output_hypernetworks_vb.pp(index),
            )?);
        }
        let iou_prediction_head = TrackerMlp::new(
            config.transformer_dim,
            config.iou_head_hidden_dim,
            num_mask_tokens,
            config.iou_head_depth,
            config.iou_prediction_use_sigmoid,
            vb.pp("iou_prediction_head"),
        )?;
        let pred_obj_score_head = if config.pred_obj_scores {
            if config.pred_obj_scores_mlp {
                Some(PredObjScoreHead::Mlp(TrackerMlp::new(
                    config.transformer_dim,
                    config.transformer_dim,
                    1,
                    3,
                    false,
                    vb.pp("pred_obj_score_head"),
                )?))
            } else {
                Some(PredObjScoreHead::Linear(linear(
                    vb.pp("pred_obj_score_head"),
                    config.transformer_dim,
                    1,
                    true,
                )?))
            }
        } else {
            None
        };
        Ok(Self {
            transformer_dim: config.transformer_dim,
            transformer,
            iou_token,
            mask_tokens,
            obj_score_token,
            output_upscaling_conv1,
            output_upscaling_ln,
            output_upscaling_conv2,
            conv_s0,
            conv_s1,
            output_hypernetworks_mlps,
            iou_prediction_head,
            pred_obj_score_head,
            num_mask_tokens,
            use_high_res_features: config.use_high_res_features,
            use_multimask_token_for_obj_ptr: config.use_multimask_token_for_obj_ptr,
            dynamic_multimask_via_stability: config.dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta: config.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh: config.dynamic_multimask_stability_thresh,
        })
    }

    fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Option<&[Tensor]>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (all_masks, all_iou_pred, mask_tokens_out, object_score_logits) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            repeat_image,
            high_res_features,
        )?;
        let masks;
        let iou_pred;
        if multimask_output {
            masks = all_masks.i((.., 1.., .., ..))?;
            iou_pred = all_iou_pred.i((.., 1..))?;
        } else if self.dynamic_multimask_via_stability {
            (masks, iou_pred) = self.dynamic_multimask_via_stability(&all_masks, &all_iou_pred)?;
        } else {
            masks = all_masks.i((.., 0..1, .., ..))?;
            iou_pred = all_iou_pred.i((.., 0..1))?;
        }

        let sam_tokens_out = if multimask_output && self.use_multimask_token_for_obj_ptr {
            mask_tokens_out.i((.., 1.., ..))?
        } else {
            mask_tokens_out.i((.., 0..1, ..))?
        };
        Ok((masks, iou_pred, sam_tokens_out, object_score_logits))
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        repeat_image: bool,
        high_res_features: Option<&[Tensor]>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut output_tokens: Vec<&Tensor> = Vec::new();
        let mut score_token_offset = 0usize;
        if let Some(obj_score_token) = &self.obj_score_token {
            output_tokens.push(obj_score_token.embeddings());
            score_token_offset = 1;
        }
        output_tokens.push(self.iou_token.embeddings());
        output_tokens.push(self.mask_tokens.embeddings());
        let output_tokens = Tensor::cat(output_tokens.as_slice(), 0)?.unsqueeze(0)?;
        let output_tokens = output_tokens.to_dtype(sparse_prompt_embeddings.dtype())?;
        let output_tokens = output_tokens.expand((
            sparse_prompt_embeddings.dim(0)?,
            output_tokens.dim(1)?,
            self.transformer_dim,
        ))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;
        let src = if repeat_image {
            repeat_interleave(image_embeddings, tokens.dim(0)?, 0)?
        } else {
            if image_embeddings.dim(0)? != tokens.dim(0)? {
                candle::bail!(
                    "tracker mask decoder expected image embedding batch {} to match token batch {}",
                    image_embeddings.dim(0)?,
                    tokens.dim(0)?
                );
            }
            image_embeddings.clone()
        };
        let src = src.broadcast_add(dense_prompt_embeddings)?;
        if image_pe.dim(0)? != 1 {
            candle::bail!(
                "tracker mask decoder expected image_pe batch dimension of 1, got {}",
                image_pe.dim(0)?
            );
        }
        let pos_src = repeat_interleave(image_pe, tokens.dim(0)?, 0)?;
        let (batch_size, channels, height, width) = src.dims4()?;
        let (hs, src_tokens) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.i((.., score_token_offset, ..))?;
        let mask_tokens_out = hs.i((
            ..,
            score_token_offset + 1..score_token_offset + 1 + self.num_mask_tokens,
            ..,
        ))?;
        let src_tokens = src_tokens
            .transpose(1, 2)?
            .reshape((batch_size, channels, height, width))?;
        let upscaled_embedding = if self.use_high_res_features {
            match high_res_features {
                Some(high_res_features) if high_res_features.len() >= 2 => {
                    let upscaled_embedding = self.output_upscaling_conv1.forward(&src_tokens)?;
                    let feat_s1 = high_res_features[1].to_dtype(upscaled_embedding.dtype())?;
                    let upscaled_embedding = upscaled_embedding.broadcast_add(&feat_s1)?;
                    let upscaled_embedding = upscaled_embedding
                        .apply(&self.output_upscaling_ln)?
                        .gelu()?;
                    let upscaled_embedding = self.output_upscaling_conv2.forward(&upscaled_embedding)?;
                    let feat_s0 = high_res_features[0].to_dtype(upscaled_embedding.dtype())?;
                    upscaled_embedding.broadcast_add(&feat_s0)?.gelu()?
                }
                _ => self
                    .output_upscaling_conv1
                    .forward(&src_tokens)?
                    .apply(&self.output_upscaling_ln)?
                    .gelu()?
                    .apply(&self.output_upscaling_conv2)?
                    .gelu()?,
            }
        } else {
            self.output_upscaling_conv1
                .forward(&src_tokens)?
                .apply(&self.output_upscaling_ln)?
                .gelu()?
                .apply(&self.output_upscaling_conv2)?
                .gelu()?
        };
        let mut hyper_in = Vec::with_capacity(self.num_mask_tokens);
        for index in 0..self.num_mask_tokens {
            hyper_in.push(
                self.output_hypernetworks_mlps[index].forward(&mask_tokens_out.i((
                    ..,
                    index,
                    ..,
                ))?)?,
            );
        }
        let hyper_in = Tensor::stack(hyper_in.as_slice(), 1)?.contiguous()?;
        let (batch_size, channels, height, width) = upscaled_embedding.dims4()?;
        let masks = hyper_in.matmul(&upscaled_embedding.reshape((
            batch_size,
            channels,
            height * width,
        ))?)?;
        let masks = masks.reshape((batch_size, self.num_mask_tokens, height, width))?;
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        let object_score_logits = match &self.pred_obj_score_head {
            Some(head) => head.forward(&hs.i((.., 0, ..))?),
            None => Tensor::ones((batch_size, 1), masks.dtype(), masks.device())? * 10f64,
        }?;
        Ok((masks, iou_pred, mask_tokens_out, object_score_logits))
    }

    fn dynamic_multimask_via_stability(
        &self,
        all_mask_logits: &Tensor,
        all_iou_scores: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let single_mask = all_mask_logits.i((.., 0..1, .., ..))?;
        let single_iou = all_iou_scores.i((.., 0..1))?;
        let stability_scores = self.stability_scores(&single_mask)?;
        let multimasks = all_mask_logits.i((.., 1.., .., ..))?;
        let multi_ious = all_iou_scores.i((.., 1..))?;
        let best_indices = multi_ious.argmax(1)?.to_vec1::<u32>()?;
        let mut best_multimasks = Vec::with_capacity(best_indices.len());
        let mut best_multi_ious = Vec::with_capacity(best_indices.len());
        for (batch_index, best_index) in best_indices.into_iter().enumerate() {
            best_multimasks.push(multimasks.i((batch_index, best_index as usize, .., ..))?);
            best_multi_ious.push(multi_ious.i((batch_index, best_index as usize))?);
        }
        let best_multimasks = Tensor::stack(best_multimasks.as_slice(), 0)?.unsqueeze(1)?;
        let best_multi_ious = Tensor::stack(best_multi_ious.as_slice(), 0)?.unsqueeze(1)?;
        let stability_ok = stability_scores.ge(self.dynamic_multimask_stability_thresh as f64)?;
        let stability_ok_masks = stability_ok
            .reshape((stability_ok.dim(0)?, 1, 1, 1))?
            .broadcast_as(best_multimasks.shape())?;
        let masks = stability_ok_masks.where_cond(&single_mask, &best_multimasks)?;
        let stability_ok_ious = stability_ok
            .reshape((stability_ok.dim(0)?, 1))?
            .broadcast_as(best_multi_ious.shape())?;
        let ious = stability_ok_ious.where_cond(&single_iou, &best_multi_ious)?;
        Ok((masks, ious))
    }

    fn stability_scores(&self, mask_logits: &Tensor) -> Result<Tensor> {
        let mask_logits = mask_logits.flatten(2, 3)?;
        let area_intersection = mask_logits
            .gt(self.dynamic_multimask_stability_delta as f64)?
            .to_dtype(DType::F32)?
            .sum(2)?;
        let area_union = mask_logits
            .gt(-(self.dynamic_multimask_stability_delta as f64))?
            .to_dtype(DType::F32)?
            .sum(2)?;
        let area_union_nonzero = area_union.gt(0f64)?;
        let safe_union =
            area_union_nonzero.where_cond(&area_union, &Tensor::ones_like(&area_union)?)?;
        let scores = area_intersection.broadcast_div(&safe_union)?;
        area_union_nonzero.where_cond(&scores, &Tensor::ones_like(&scores)?)
    }
}

#[derive(Debug)]
pub struct Sam3TrackerModel {
    config: Sam3TrackerConfig,
    vision_trunk: Option<Sam3ViTDetTrunk>,
    vision_neck: Option<Sam3DualViTDetNeck>,
    mask_downsample: Conv2d,
    sam_prompt_encoder: PromptEncoder,
    sam_mask_decoder: Sam3TrackerMaskDecoder,
    obj_ptr_proj: TrackerMlp,
    obj_ptr_tpos_proj: Linear,
    maskmem_tpos_enc: Tensor,
    no_mem_embed: Tensor,
    no_mem_pos_enc: Tensor,
    no_obj_ptr: Tensor,
    no_obj_embed_spatial: Tensor,
}

impl Sam3TrackerModel {
    pub fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        let (vision_trunk, vision_neck) = if config.predictor.with_backbone {
            (
                Some(Sam3ViTDetTrunk::new(
                    &Config::default().vision,
                    vb.pp("backbone").pp("vision_backbone").pp("trunk"),
                )?),
                Some(Sam3DualViTDetNeck::new(
                    &Config::default().neck,
                    vb.pp("backbone").pp("vision_backbone"),
                )?),
            )
        } else {
            (None, None)
        };
        let mask_downsample = candle_nn::conv2d(
            1,
            1,
            4,
            Conv2dConfig {
                stride: 4,
                ..Default::default()
            },
            vb.pp("mask_downsample"),
        )?;
        let sam_prompt_encoder = PromptEncoder::new(
            config.prompt_encoder.embed_dim,
            (
                config.prompt_encoder.image_embedding_size[0],
                config.prompt_encoder.image_embedding_size[1],
            ),
            (
                config.prompt_encoder.input_image_size[0],
                config.prompt_encoder.input_image_size[1],
            ),
            config.prompt_encoder.mask_in_chans,
            vb.pp("sam_prompt_encoder"),
        )?;
        let sam_mask_decoder =
            Sam3TrackerMaskDecoder::new(&config.mask_decoder, vb.pp("sam_mask_decoder"))?;
        let obj_ptr_proj = TrackerMlp::new(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            3,
            false,
            vb.pp("obj_ptr_proj"),
        )?;
        let obj_ptr_tpos_proj = linear(
            vb.pp("obj_ptr_tpos_proj"),
            config.hidden_dim,
            config.memory_dim,
            true,
        )?;
        Ok(Self {
            config: config.clone(),
            vision_trunk,
            vision_neck,
            mask_downsample,
            sam_prompt_encoder,
            sam_mask_decoder,
            obj_ptr_proj,
            obj_ptr_tpos_proj,
            maskmem_tpos_enc: vb.get(&config.shapes.maskmem_tpos_enc_shape, "maskmem_tpos_enc")?,
            no_mem_embed: vb.get(&config.shapes.no_mem_embed_shape, "no_mem_embed")?,
            no_mem_pos_enc: vb.get(&config.shapes.no_mem_pos_enc_shape, "no_mem_pos_enc")?,
            no_obj_ptr: vb.get(&config.shapes.no_obj_ptr_shape, "no_obj_ptr")?,
            no_obj_embed_spatial: vb.get(
                &config.shapes.no_obj_embed_spatial_shape,
                "no_obj_embed_spatial",
            )?,
        })
    }

    pub fn from_checkpoint_source(
        sam3_config: &Config,
        checkpoint: &Sam3CheckpointSource,
        dtype: DType,
        device: &candle::Device,
    ) -> Result<Self> {
        let tracker_config = Sam3TrackerConfig::from_sam3_config(sam3_config);
        Self::new(
            &tracker_config,
            checkpoint.load_tracker_var_builder(dtype, device)?,
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

    fn get_tpos_enc(
        &self,
        rel_pos_list: &[i64],
        device: &Device,
        max_abs_pos: Option<usize>,
        dummy: bool,
    ) -> Result<Tensor> {
        if dummy {
            return Tensor::zeros((rel_pos_list.len(), self.config.memory_dim), DType::F32, device);
        }

        let t_diff_max = max_abs_pos
            .map(|value| value.saturating_sub(1).max(1))
            .unwrap_or(1) as f64;
        let pos_inds = Tensor::from_vec(
            rel_pos_list
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            rel_pos_list.len(),
            device,
        )?;
        let pos_inds = pos_inds.broadcast_div(&Tensor::new(t_diff_max as f32, device)?)?;
        let pos_enc = get_1d_sine_pe(&pos_inds, self.config.hidden_dim)?;
        self.obj_ptr_tpos_proj.forward(&pos_enc)
    }

    pub fn encode_image_features(&self, image: &Tensor) -> Result<VisualBackboneOutput> {
        let vision_trunk = self.vision_trunk.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "tracker image-feature path is unavailable because predictor.with_backbone=false"
                    .to_owned(),
            )
        })?;
        let vision_neck = self.vision_neck.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "tracker image-feature path is unavailable because predictor.with_backbone=false"
                    .to_owned(),
            )
        })?;
        let image = match image.rank() {
            3 => image.unsqueeze(0)?,
            4 => image.clone(),
            rank => {
                candle::bail!("sam3 tracker image encoder expects CHW or BCHW input, got rank {rank}")
            }
        };
        let trunk = vision_trunk.forward(&image)?;
        vision_neck.forward(&trunk)
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
        is_conditioning_frame: bool,
        reverse: bool,
        _use_prev_mem_frame: bool,
        run_mem_encoder: bool,
    ) -> Result<TrackerStepOutput> {
        if reverse {
            candle::bail!(
                "SAM3 tracker strict port currently supports prompt-frame forward tracking only; reverse tracking lands in a later step."
            );
        }
        if !_history.is_empty() {
            candle::bail!(
                "SAM3 tracker strict port currently supports prompt frames without history only; memory-conditioned tracking lands in a later step."
            );
        }
        if run_mem_encoder {
            candle::bail!(
                "SAM3 tracker strict port currently supports run_mem_encoder=false only; memory writes land in a later step."
            );
        }
        if _visual.backbone_fpn.is_empty() {
            candle::bail!("tracker requires at least one visual feature level")
        }
        if _visual.vision_pos_enc.is_empty() {
            candle::bail!("tracker requires at least one visual position-encoding level")
        }
        if _mask_input.is_some() && (_point_coords.is_some() || _boxes_xyxy.is_some()) {
            candle::bail!(
                "SAM3 tracker strict port currently supports either mask prompts or point/box prompts, not both at once."
            );
        }
        let compute_dtype = self.no_obj_ptr.dtype();
        let backbone_features = _visual
            .backbone_fpn
            .last()
            .expect("checked non-empty above")
            .to_dtype(compute_dtype)?;
        let prepared_high_res_features = if _visual.backbone_fpn.len() > 1 {
            Some(self.prepare_high_res_features(
                &_visual.backbone_fpn[.._visual.backbone_fpn.len() - 1],
            )?)
        } else {
            None
        };
        let high_res_features = prepared_high_res_features.as_deref();
        if let Some(mask_input) = _mask_input {
            let state = self.use_mask_as_output(
                &backbone_features,
                high_res_features,
                mask_input,
                is_conditioning_frame,
            )?;
            return Ok(TrackerStepOutput {
                state,
                prompt_frame_indices: if is_conditioning_frame {
                    vec![_frame_idx]
                } else {
                    Vec::new()
                },
                memory_frame_indices: Vec::new(),
            });
        }
        let point_prompt = self.prepare_point_prompt(
            _point_coords,
            _point_labels,
            _boxes_xyxy,
            backbone_features.device(),
        )?;
        let point_count = point_prompt
            .as_ref()
            .map(|(_, labels)| labels.dim(1).unwrap_or(0))
            .unwrap_or(0);
        let multimask_output = self.use_multimask(is_conditioning_frame, point_count);
        let state = self.forward_sam_heads(
            &backbone_features,
            point_prompt.as_ref(),
            None,
            high_res_features,
            multimask_output,
            is_conditioning_frame,
        )?;
        Ok(TrackerStepOutput {
            state,
            prompt_frame_indices: if is_conditioning_frame {
                vec![_frame_idx]
            } else {
                Vec::new()
            },
            memory_frame_indices: Vec::new(),
        })
    }

    fn prepare_point_prompt(
        &self,
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes_xyxy: Option<&Tensor>,
        device: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let point_coords = match point_coords {
            Some(coords) => normalize_point_coords(coords, device)?,
            None => Tensor::zeros((1, 0, 2), DType::F32, device)?,
        };
        let point_labels = match point_labels {
            Some(labels) => normalize_point_labels(labels, device)?,
            None => Tensor::zeros((1, 0), DType::F32, device)?,
        };
        let (point_coords, point_labels) = if let Some(boxes_xyxy) = boxes_xyxy {
            let box_coords = normalize_boxes_as_points(boxes_xyxy, device)?;
            let batch_size = box_coords.dim(0)?;
            let box_labels =
                Tensor::from_vec(vec![2f32, 3f32].repeat(batch_size), (batch_size, 2), device)?;
            (
                Tensor::cat(&[&box_coords, &point_coords], 1)?,
                Tensor::cat(&[&box_labels, &point_labels], 1)?,
            )
        } else {
            (point_coords, point_labels)
        };

        if point_coords.dim(1)? == 0 {
            Ok(None)
        } else {
            Ok(Some((point_coords, point_labels)))
        }
    }

    fn use_multimask(&self, is_init_cond_frame: bool, point_count: usize) -> bool {
        self.config.multimask_output_in_sam
            && (is_init_cond_frame || self.config.multimask_output_for_tracking)
            && (self.config.multimask_min_pt_num..=self.config.multimask_max_pt_num)
                .contains(&point_count)
    }

    fn prepare_high_res_features(&self, high_res_features: &[Tensor]) -> Result<Vec<Tensor>> {
        if high_res_features.len() < 2 {
            candle::bail!(
                "tracker expected at least two high-resolution feature levels, got {}",
                high_res_features.len()
            );
        }
        let feat_s0 = &high_res_features[0];
        let feat_s1 = &high_res_features[1];
        let compute_dtype = self.no_obj_ptr.dtype();
        let projected_s0 = self.config.mask_decoder.transformer_dim / 8;
        let projected_s1 = self.config.mask_decoder.transformer_dim / 4;
        let (_, channels_s0, _, _) = feat_s0.dims4()?;
        let (_, channels_s1, _, _) = feat_s1.dims4()?;
        if channels_s0 == projected_s0 && channels_s1 == projected_s1 {
            return Ok(vec![
                feat_s0.to_dtype(compute_dtype)?,
                feat_s1.to_dtype(compute_dtype)?,
            ]);
        }
        if channels_s0 == self.config.hidden_dim && channels_s1 == self.config.hidden_dim {
            let conv_s0 = self
                .sam_mask_decoder
                .conv_s0
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("tracker high-res projection conv_s0 missing".into()))?;
            let conv_s1 = self
                .sam_mask_decoder
                .conv_s1
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("tracker high-res projection conv_s1 missing".into()))?;
            return Ok(vec![
                feat_s0.apply(conv_s0)?.to_dtype(compute_dtype)?,
                feat_s1.apply(conv_s1)?.to_dtype(compute_dtype)?,
            ]);
        }
        candle::bail!(
            "unexpected tracker high-res feature channel contract: s0={}, s1={}, expected projected [{projected_s0}, {projected_s1}] or hidden_dim {}",
            channels_s0,
            channels_s1,
            self.config.hidden_dim
        );
    }

    #[cfg(test)]
    pub(crate) fn prepare_high_res_features_for_test(
        &self,
        high_res_features: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        self.prepare_high_res_features(high_res_features)
    }

    fn forward_sam_heads(
        &self,
        backbone_features: &Tensor,
        point_prompt: Option<&(Tensor, Tensor)>,
        mask_inputs: Option<&Tensor>,
        high_res_features: Option<&[Tensor]>,
        multimask_output: bool,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState> {
        let batch_size = backbone_features.dim(0)?;
        let device = backbone_features.device();
        let (sam_point_coords, sam_point_labels) = match point_prompt {
            Some((coords, labels)) => (coords.clone(), labels.clone()),
            None => (
                Tensor::zeros((batch_size, 1, 2), DType::F32, device)?,
                (Tensor::ones((batch_size, 1), DType::F32, device)? * -1f64)?,
            ),
        };
        let sam_mask_prompt = match mask_inputs {
            Some(mask_inputs) => {
                let mask_inputs = normalize_mask_prompt(mask_inputs, device)?;
                let (_, _, height, width) = mask_inputs.dims4()?;
                if [height, width] != self.config.prompt_encoder.mask_input_size {
                    Some(mask_inputs.upsample_bilinear2d(
                        self.config.prompt_encoder.mask_input_size[0],
                        self.config.prompt_encoder.mask_input_size[1],
                        false,
                    )?)
                } else {
                    Some(mask_inputs)
                }
            }
            None => None,
        };
        let (sparse_embeddings, dense_embeddings) = self.sam_prompt_encoder.forward(
            Some((&sam_point_coords, &sam_point_labels)),
            None,
            sam_mask_prompt.as_ref(),
        )?;
        let backbone_dtype = backbone_features.dtype();
        let sparse_embeddings = sparse_embeddings.to_dtype(backbone_dtype)?;
        let dense_embeddings = dense_embeddings.to_dtype(backbone_dtype)?;
        let image_pe = self
            .sam_prompt_encoder
            .get_dense_pe()?
            .to_dtype(backbone_dtype)?;
        let (low_res_multimasks, ious, sam_output_tokens, object_score_logits) =
            self.sam_mask_decoder.forward(
                backbone_features,
                &image_pe,
                &sparse_embeddings,
                &dense_embeddings,
                multimask_output,
                false,
                high_res_features,
            )?;
        let object_present = object_score_logits.gt(0f64)?;
        let gated_low_res_multimasks = object_present
            .reshape((batch_size, 1, 1, 1))?
            .broadcast_as(low_res_multimasks.shape())?
            .where_cond(
                &low_res_multimasks,
                &Tensor::full(NO_OBJ_SCORE as f32, low_res_multimasks.shape(), device)?,
            )?;
        let high_res_multimasks = gated_low_res_multimasks.upsample_bilinear2d(
            self.config.image_size,
            self.config.image_size,
            false,
        )?;
        let (low_res_masks, high_res_masks, sam_output_token) = if multimask_output {
            let best_iou_indices = ious.argmax(1)?.to_vec1::<u32>()?;
            let mut low_res_masks = Vec::with_capacity(best_iou_indices.len());
            let mut high_res_masks = Vec::with_capacity(best_iou_indices.len());
            let mut sam_output_tokens_best = Vec::with_capacity(best_iou_indices.len());
            for (batch_index, best_index) in best_iou_indices.into_iter().enumerate() {
                let best_index = best_index as usize;
                low_res_masks.push(gated_low_res_multimasks.i((
                    batch_index,
                    best_index,
                    ..,
                    ..,
                ))?);
                high_res_masks.push(high_res_multimasks.i((batch_index, best_index, .., ..))?);
                sam_output_tokens_best.push(if sam_output_tokens.dim(1)? > 1 {
                    sam_output_tokens.i((batch_index, best_index, ..))?
                } else {
                    sam_output_tokens.i((batch_index, 0, ..))?
                });
            }
            (
                Tensor::stack(low_res_masks.as_slice(), 0)?.unsqueeze(1)?,
                Tensor::stack(high_res_masks.as_slice(), 0)?.unsqueeze(1)?,
                Tensor::stack(sam_output_tokens_best.as_slice(), 0)?,
            )
        } else {
            (
                gated_low_res_multimasks.clone(),
                high_res_multimasks,
                sam_output_tokens.i((.., 0, ..))?,
            )
        };
        let obj_ptr = self.obj_ptr_proj.forward(&sam_output_token)?;
        let object_present_for_ptr = object_present
            .broadcast_as(obj_ptr.shape())?
            .where_cond(&obj_ptr, &self.no_obj_ptr.broadcast_as(obj_ptr.shape())?)?;
        Ok(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores: ious,
            obj_ptr: object_present_for_ptr,
            object_score_logits,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame,
        })
    }

    fn use_mask_as_output(
        &self,
        backbone_features: &Tensor,
        high_res_features: Option<&[Tensor]>,
        mask_inputs: &Tensor,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState> {
        let device = backbone_features.device();
        let mask_inputs = normalize_mask_prompt(mask_inputs, device)?;
        let mask_inputs_float = mask_inputs.to_dtype(DType::F32)?;
        let high_res_masks = mask_inputs_float.affine(20.0, -10.0)?;
        let mask_input_low_res_size = (self.input_mask_size() / self.config.backbone_stride) * 4;
        let low_res_masks = resize_bilinear2d_antialias(
            &high_res_masks,
            mask_input_low_res_size,
            mask_input_low_res_size,
        )?;
        let iou_scores = Tensor::ones((mask_inputs_float.dim(0)?, 1), DType::F32, device)?;
        let mask_prompt = self.mask_downsample.forward(&mask_inputs_float)?;
        let prepared_high_res_features = match high_res_features {
            Some(high_res_features) => Some(self.prepare_high_res_features(high_res_features)?),
            None => None,
        };
        let state = self.forward_sam_heads(
            backbone_features,
            None,
            Some(&mask_prompt),
            prepared_high_res_features.as_deref(),
            false,
            is_cond_frame,
        )?;
        let object_present = mask_inputs_float
            .flatten(1, 3)?
            .gt(0f64)?
            .to_dtype(DType::F32)?
            .sum(1)?
            .gt(0f64)?
            .unsqueeze(1)?;
        let object_score_logits = object_present.to_dtype(DType::F32)?.affine(20.0, -10.0)?;
        let obj_ptr = object_present
            .broadcast_as(state.obj_ptr.shape())?
            .where_cond(
                &state.obj_ptr,
                &self.no_obj_ptr.broadcast_as(state.obj_ptr.shape())?,
            )?;
        Ok(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores,
            obj_ptr,
            object_score_logits,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame,
        })
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

fn get_1d_sine_pe(pos_inds: &Tensor, dim: usize) -> Result<Tensor> {
    if dim % 2 != 0 {
        candle::bail!("tracker temporal position encoding requires even dim, got {dim}");
    }
    let device = pos_inds.device();
    let dtype = pos_inds.dtype();
    let pe_dim = dim / 2;
    let mut dim_t = Vec::with_capacity(pe_dim);
    for idx in 0..pe_dim {
        let exponent = 2.0 * (idx / 2) as f32 / pe_dim as f32;
        dim_t.push(10_000f32.powf(exponent));
    }
    let dim_t = Tensor::from_vec(dim_t, pe_dim, device)?.to_dtype(dtype)?;
    let pos_embed = pos_inds.unsqueeze(1)?.broadcast_div(&dim_t)?;
    let sin = pos_embed.sin()?;
    let cos = pos_embed.cos()?;
    Tensor::cat(&[&sin, &cos], 1)
}

pub(crate) fn resize_bilinear2d_antialias(
    input: &Tensor,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    let input_cpu = input.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let (batch, channels, in_h, in_w) = input_cpu.dims4()?;
    let input_vec = input_cpu.flatten_all()?.to_vec1::<f32>()?;
    let width_weights = antialias_linear_weights(in_w, out_w);
    let height_weights = antialias_linear_weights(in_h, out_h);
    let mut horizontal = vec![0.0f32; batch * channels * in_h * out_w];
    let mut output = vec![0.0f32; batch * channels * out_h * out_w];
    let input_stride_c = in_h * in_w;
    let input_stride_b = channels * input_stride_c;
    let horizontal_stride_c = in_h * out_w;
    let horizontal_stride_b = channels * horizontal_stride_c;
    let output_stride_c = out_h * out_w;
    let output_stride_b = channels * output_stride_c;

    for b in 0..batch {
        for c in 0..channels {
            let input_base = b * input_stride_b + c * input_stride_c;
            let horizontal_base = b * horizontal_stride_b + c * horizontal_stride_c;
            let output_base = b * output_stride_b + c * output_stride_c;
            for y in 0..in_h {
                let row_offset = input_base + y * in_w;
                let horizontal_row_offset = horizontal_base + y * out_w;
                for (out_x, weights) in width_weights.iter().enumerate() {
                    let mut value = 0.0f32;
                    for (src_x, weight) in weights {
                        value += input_vec[row_offset + *src_x] * *weight;
                    }
                    horizontal[horizontal_row_offset + out_x] = value;
                }
            }
            for (out_y, weights) in height_weights.iter().enumerate() {
                let output_row_offset = output_base + out_y * out_w;
                for out_x in 0..out_w {
                    let mut value = 0.0f32;
                    for (src_y, weight) in weights {
                        value += horizontal[horizontal_base + *src_y * out_w + out_x] * *weight;
                    }
                    output[output_row_offset + out_x] = value;
                }
            }
        }
    }

    Tensor::from_vec(output, (batch, channels, out_h, out_w), &Device::Cpu)?.to_device(input.device())
}

fn antialias_linear_weights(input_size: usize, output_size: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input_size as f32 / output_size as f32;
    let support = scale.max(1.0);
    let radius = support;
    let mut all_weights = Vec::with_capacity(output_size);
    for out_idx in 0..output_size {
        let center = scale * (out_idx as f32 + 0.5) - 0.5;
        let xmin = (center - radius).floor() as isize;
        let xmax = (center + radius).ceil() as isize;
        let mut weights = Vec::new();
        let mut weight_sum = 0.0f32;
        for src_idx in xmin..=xmax {
            let distance = (src_idx as f32 - center) / support;
            let weight = (1.0 - distance.abs()).max(0.0) / support;
            if weight == 0.0 {
                continue;
            }
            let clamped = src_idx.clamp(0, input_size.saturating_sub(1) as isize) as usize;
            weights.push((clamped, weight));
            weight_sum += weight;
        }
        if weight_sum > 0.0 {
            for (_, weight) in weights.iter_mut() {
                *weight /= weight_sum;
            }
        }
        all_weights.push(weights);
    }
    all_weights
}

fn normalize_point_coords(coords: &Tensor, device: &Device) -> Result<Tensor> {
    let coords = coords.to_device(device)?.to_dtype(DType::F32)?;
    match coords.rank() {
        2 => coords.unsqueeze(0),
        3 => Ok(coords),
        rank => candle::bail!("tracker point coords must have rank 2 or 3, got {rank}"),
    }
}

fn normalize_point_labels(labels: &Tensor, device: &Device) -> Result<Tensor> {
    let labels = labels.to_device(device)?.to_dtype(DType::F32)?;
    match labels.rank() {
        1 => labels.unsqueeze(0),
        2 => Ok(labels),
        rank => candle::bail!("tracker point labels must have rank 1 or 2, got {rank}"),
    }
}

fn normalize_boxes_as_points(boxes_xyxy: &Tensor, device: &Device) -> Result<Tensor> {
    let boxes_xyxy = boxes_xyxy.to_device(device)?.to_dtype(DType::F32)?;
    match boxes_xyxy.rank() {
        1 => boxes_xyxy.reshape((1, 2, 2)),
        2 => boxes_xyxy.reshape((boxes_xyxy.dim(0)?, 2, 2)),
        3 => Ok(boxes_xyxy),
        rank => candle::bail!("tracker boxes must have rank 1, 2, or 3, got {rank}"),
    }
}

fn normalize_mask_prompt(mask: &Tensor, device: &Device) -> Result<Tensor> {
    let mask = mask.to_device(device)?.to_dtype(DType::F32)?;
    match mask.rank() {
        2 => mask.unsqueeze(0)?.unsqueeze(0),
        3 => mask.unsqueeze(1),
        4 => Ok(mask),
        rank => candle::bail!("tracker mask input must have rank 2, 3, or 4, got {rank}"),
    }
}

fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let xs = xs.unsqueeze(dim + 1)?;
    let mut dims = xs.dims().to_vec();
    dims[dim + 1] = repeats;
    xs.broadcast_as(dims)?.flatten(dim, dim + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
    };

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
        let feat0 = Tensor::zeros((1, 32, 16, 16), DType::F32, device)?;
        let feat1 = Tensor::zeros((1, 32, 8, 8), DType::F32, device)?;
        let feat2 = Tensor::zeros((1, 32, 4, 4), DType::F32, device)?;
        let pos0 = Tensor::zeros((1, 32, 16, 16), DType::F32, device)?;
        let pos1 = Tensor::zeros((1, 32, 8, 8), DType::F32, device)?;
        let pos2 = Tensor::zeros((1, 32, 4, 4), DType::F32, device)?;
        Ok(VisualBackboneOutput {
            backbone_fpn: vec![feat0, feat1, feat2],
            vision_pos_enc: vec![pos0, pos1, pos2],
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
        PointSingleClick,
        PointMultiClick,
        PointAllPoints,
        MaskDirect,
        MultimaskDisabledTracking,
        MultimaskDisabledSam,
    }

    impl TrackerFixtureBundle {
        fn debug_dir(self) -> &'static str {
            match self {
                Self::Default => "../candle-examples/examples/sam3/reference_video_box_debug/debug",
                Self::TemporalDisambiguation => {
                    "../candle-examples/examples/sam3/reference_video_box_debug_temporal_disambiguation/debug"
                }
                Self::PointSingleClick => {
                    "../candle-examples/examples/sam3/reference_video_point_debug_single_click/debug"
                }
                Self::PointMultiClick => {
                    "../candle-examples/examples/sam3/reference_video_point_debug_multi_click/debug"
                }
                Self::PointAllPoints => {
                    "../candle-examples/examples/sam3/reference_video_point_debug_all_points/debug"
                }
                Self::MaskDirect => {
                    "../candle-examples/examples/sam3/reference_video_mask_debug/debug"
                }
                Self::MultimaskDisabledTracking => {
                    "../candle-examples/examples/sam3/reference_video_multimask_disabled_tracking_debug/debug"
                }
                Self::MultimaskDisabledSam => {
                    "../candle-examples/examples/sam3/reference_video_multimask_disabled_sam_debug/debug"
                }
            }
        }
    }

    fn tracker_fixture_dir(bundle: TrackerFixtureBundle) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(bundle.debug_dir())
    }

    fn tracker_fixture_tensor_path(bundle: TrackerFixtureBundle) -> PathBuf {
        tracker_fixture_dir(bundle).join("internal_fixtures.safetensors")
    }

    fn load_tracker_internal_manifest(
        bundle: TrackerFixtureBundle,
    ) -> Result<TrackerInternalManifest> {
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

    fn load_tracker_fixture_tensor(bundle: TrackerFixtureBundle, key: &str) -> Result<Tensor> {
        use candle::safetensors::Load;

        let path = tracker_fixture_tensor_path(bundle);
        let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(&path) }.map_err(
            |err| {
                candle::Error::Msg(format!(
                    "failed to mmap tracker fixture tensors {}: {err}",
                    path.display()
                ))
            },
        )?;
        tensors
            .get(key)
            .map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to read tensor `{key}` from tracker fixture {}: {err}",
                    path.display()
                ))
            })?
            .load(&candle::Device::Cpu)
    }

    fn tracker_test_checkpoint_path() -> Option<PathBuf> {
        let env_path = std::env::var_os("SAM3_TEST_CHECKPOINT")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("SAM3_TEST_CHECKPOINT_DIR").map(PathBuf::from));
        let mut candidates = Vec::new();
        if let Some(path) = env_path {
            candidates.push(path);
        }
        candidates.push(PathBuf::from("/home/dnorthover/extcode/hf_sam3"));
        candidates.push(PathBuf::from("/home/dnorthover/extcode/hf_sam3/sam3.pt"));
        candidates.into_iter().find_map(|path| {
            if path.is_dir() {
                let file = path.join("sam3.pt");
                file.exists().then_some(file)
            } else if path.exists() {
                Some(path)
            } else {
                None
            }
        })
    }

    fn load_runtime_tracker_model_from_checkpoint() -> Result<Option<Sam3TrackerModel>> {
        let Some(checkpoint_path) = tracker_test_checkpoint_path() else {
            return Ok(None);
        };
        let config = Config::default();
        Sam3TrackerModel::from_checkpoint_source(
            &config,
            &Sam3CheckpointSource::upstream_pth(checkpoint_path),
            DType::F32,
            &candle::Device::Cpu,
        )
        .map(Some)
    }

    fn tracker_runtime_config_from_fixture_manifest(
        manifest: &TrackerInternalManifest,
    ) -> Sam3TrackerConfig {
        let fixture = &manifest.tracker_config;
        let predictor = &manifest.predictor_config;
        let mut config = Sam3TrackerConfig::build_tracker(fixture.use_memory_selection);
        config.multimask_output_in_sam = fixture.multimask_output_in_sam;
        config.multimask_output_for_tracking = fixture.multimask_output_for_tracking;
        config.multimask_min_pt_num = fixture.multimask_min_pt_num;
        config.multimask_max_pt_num = fixture.multimask_max_pt_num;
        config.mask_decoder.dynamic_multimask_via_stability =
            fixture.sam_mask_decoder_extra_args.dynamic_multimask_via_stability;
        config.mask_decoder.dynamic_multimask_stability_delta =
            fixture.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta;
        config.mask_decoder.dynamic_multimask_stability_thresh =
            fixture.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh;
        config.predictor.with_backbone = false;
        config.predictor.forward_backbone_per_frame_for_eval =
            fixture.forward_backbone_per_frame_for_eval;
        config.predictor.trim_past_non_cond_mem_for_eval =
            fixture.trim_past_non_cond_mem_for_eval;
        config.predictor.offload_output_to_cpu_for_eval =
            fixture.offload_output_to_cpu_for_eval;
        config.predictor.clear_non_cond_mem_around_input =
            predictor.clear_non_cond_mem_around_input;
        config.predictor.clear_non_cond_mem_for_multi_obj =
            predictor.clear_non_cond_mem_for_multi_obj;
        config.predictor.fill_hole_area = predictor.fill_hole_area;
        config.predictor.always_start_from_first_ann_frame =
            predictor.always_start_from_first_ann_frame;
        config.predictor.max_point_num_in_prompt_enc = predictor.max_point_num_in_prompt_enc;
        config.predictor.non_overlap_masks_for_output = predictor.non_overlap_masks_for_output;
        config.predictor.iter_use_prev_mask_pred = predictor.iter_use_prev_mask_pred;
        config.predictor.add_all_frames_to_correct_as_cond =
            predictor.add_all_frames_to_correct_as_cond;
        config.predictor.use_prev_mem_frame = predictor.use_prev_mem_frame;
        config.predictor.use_stateless_refinement = predictor.use_stateless_refinement;
        config.predictor.refinement_detector_cond_frame_removal_window =
            predictor.refinement_detector_cond_frame_removal_window;
        config.predictor.hotstart_delay = predictor.hotstart_delay;
        config.predictor.hotstart_unmatch_thresh = predictor.hotstart_unmatch_thresh;
        config.predictor.hotstart_dup_thresh = predictor.hotstart_dup_thresh;
        config.predictor.masklet_confirmation_enable =
            predictor.masklet_confirmation_enable;
        config.predictor.masklet_confirmation_consecutive_det_thresh =
            predictor.masklet_confirmation_consecutive_det_thresh;
        config.predictor.compile_all_components = predictor.compile_model;
        config
    }

    fn load_runtime_tracker_model_from_bundle(
        bundle: TrackerFixtureBundle,
    ) -> Result<Option<Sam3TrackerModel>> {
        let Some(checkpoint_path) = tracker_test_checkpoint_path() else {
            return Ok(None);
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let config = tracker_runtime_config_from_fixture_manifest(&manifest);
        let checkpoint = Sam3CheckpointSource::upstream_pth(checkpoint_path);
        let vb = checkpoint.load_tracker_var_builder(DType::F32, &candle::Device::Cpu)?;
        Sam3TrackerModel::new(&config, vb).map(Some)
    }

    fn build_fixture_visual_output(
        bundle: TrackerFixtureBundle,
        forward_stage: &TrackerInternalRecord,
    ) -> Result<VisualBackboneOutput> {
        let high_res_0 = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["high_res_features.0"].as_str(),
        )?;
        let high_res_1 = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["high_res_features.1"].as_str(),
        )?;
        let backbone = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["backbone_features"].as_str(),
        )?;
        let pos0 = Tensor::zeros(high_res_0.shape(), high_res_0.dtype(), &candle::Device::Cpu)?;
        let pos1 = Tensor::zeros(high_res_1.shape(), high_res_1.dtype(), &candle::Device::Cpu)?;
        let pos2 = Tensor::zeros(backbone.shape(), backbone.dtype(), &candle::Device::Cpu)?;
        Ok(VisualBackboneOutput {
            backbone_fpn: vec![high_res_0, high_res_1, backbone],
            vision_pos_enc: vec![pos0, pos1, pos2],
            sam2_backbone_fpn: None,
            sam2_pos_enc: None,
        })
    }

    fn assert_tensor_close(label: &str, actual: &Tensor, expected: &Tensor, atol: f32) -> Result<()> {
        if actual.shape() != expected.shape() {
            candle::bail!(
                "{label} shape mismatch: actual {:?}, expected {:?}",
                actual.shape().dims(),
                expected.shape().dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        let max_abs_diff = actual
            .broadcast_sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        if max_abs_diff > atol {
            candle::bail!(
                "{label} max abs diff {max_abs_diff:.6} exceeded tolerance {atol:.6}"
            );
        }
        Ok(())
    }

    fn assert_prompt_frame_point_fixture_matches(
        bundle: TrackerFixtureBundle,
        expected_point_count: usize,
        low_res_mask_atol: f32,
        high_res_mask_atol: f32,
        iou_atol: f32,
        obj_ptr_atol: f32,
        object_score_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_checkpoint()? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let forward_stage = tracker_record(&manifest, 0, "forward_sam_heads")?;
        let track_stage = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(
            track_stage.metadata["point_input_count"].as_u64(),
            Some(expected_point_count as u64)
        );
        let visual = build_fixture_visual_output(bundle, forward_stage)?;
        let point_coords =
            load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys["point_inputs.point_coords"].as_str())?;
        let point_labels =
            load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys["point_inputs.point_labels"].as_str())?;
        let actual = model.track_frame(
            &visual,
            0,
            30,
            Some(&point_coords),
            Some(&point_labels),
            None,
            None,
            &BTreeMap::new(),
            true,
            false,
            false,
            false,
        )?;
        let expected_low_res_masks = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["forward_sam_heads_output.low_res_masks"].as_str(),
        )?;
        let expected_high_res_masks = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["forward_sam_heads_output.high_res_masks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["forward_sam_heads_output.ious"].as_str(),
        )?;
        let expected_obj_ptr = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["forward_sam_heads_output.obj_ptr"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["forward_sam_heads_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "prompt point low_res_masks",
            &actual.state.low_res_masks,
            &expected_low_res_masks,
            low_res_mask_atol,
        )?;
        assert_tensor_close(
            "prompt point high_res_masks",
            &actual.state.high_res_masks,
            &expected_high_res_masks,
            high_res_mask_atol,
        )?;
        assert_tensor_close(
            "prompt point iou_scores",
            &actual.state.iou_scores,
            &expected_ious,
            iou_atol,
        )?;
        assert_tensor_close(
            "prompt point obj_ptr",
            &actual.state.obj_ptr,
            &expected_obj_ptr,
            obj_ptr_atol,
        )?;
        assert_tensor_close(
            "prompt point object_score_logits",
            &actual.state.object_score_logits,
            &expected_object_score_logits,
            object_score_atol,
        )?;
        Ok(())
    }

    fn assert_mask_decoder_fixture_matches(
        bundle: TrackerFixtureBundle,
        low_res_atol: f32,
        iou_atol: f32,
        token_atol: f32,
        object_score_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let stage = tracker_record(&manifest, 0, "sam_mask_decoder")?;
        let image_embeddings = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_inputs.image_embeddings"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let image_pe = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_inputs.image_pe"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let sparse_prompt_embeddings = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_inputs.sparse_prompt_embeddings"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let dense_prompt_embeddings = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_inputs.dense_prompt_embeddings"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let high_res_features = if stage
            .tensor_keys
            .contains_key("mask_decoder_inputs.high_res_features.0")
        {
            Some(vec![
                load_tracker_fixture_tensor(
                    bundle,
                    stage.tensor_keys["mask_decoder_inputs.high_res_features.0"].as_str(),
                )?
                .to_dtype(DType::F32)?,
                load_tracker_fixture_tensor(
                    bundle,
                    stage.tensor_keys["mask_decoder_inputs.high_res_features.1"].as_str(),
                )?
                .to_dtype(DType::F32)?,
            ])
        } else {
            None
        };
        let (low_res_multimasks, ious, sam_output_tokens, object_score_logits) = model
            .sam_mask_decoder
            .forward(
                &image_embeddings,
                &image_pe,
                &sparse_prompt_embeddings,
                &dense_prompt_embeddings,
                stage.metadata["multimask_output"].as_bool().unwrap_or(false),
                stage.metadata["repeat_image"].as_bool().unwrap_or(false),
                high_res_features.as_deref(),
            )?;
        let expected_low_res_multimasks = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_output.low_res_multimasks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_output.ious"].as_str(),
        )?;
        let expected_sam_output_tokens = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_output.sam_output_tokens"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["mask_decoder_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "mask decoder low_res_multimasks",
            &low_res_multimasks,
            &expected_low_res_multimasks,
            low_res_atol,
        )?;
        assert_tensor_close("mask decoder ious", &ious, &expected_ious, iou_atol)?;
        assert_tensor_close(
            "mask decoder sam_output_tokens",
            &sam_output_tokens,
            &expected_sam_output_tokens,
            token_atol,
        )?;
        assert_tensor_close(
            "mask decoder object_score_logits",
            &object_score_logits,
            &expected_object_score_logits,
            object_score_atol,
        )?;
        Ok(())
    }

    fn assert_forward_sam_heads_fixture_matches(
        bundle: TrackerFixtureBundle,
        low_res_mask_atol: f32,
        high_res_mask_atol: f32,
        iou_atol: f32,
        obj_ptr_atol: f32,
        object_score_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let stage = tracker_record(&manifest, 0, "forward_sam_heads")?;
        let backbone_features = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["backbone_features"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let point_prompt = if stage.metadata["has_point_inputs"].as_bool().unwrap_or(false) {
            let point_coords = normalize_point_coords(
                &load_tracker_fixture_tensor(
                    bundle,
                    stage.tensor_keys["point_inputs.point_coords"].as_str(),
                )?,
                &candle::Device::Cpu,
            )?;
            let point_labels = normalize_point_labels(
                &load_tracker_fixture_tensor(
                    bundle,
                    stage.tensor_keys["point_inputs.point_labels"].as_str(),
                )?,
                &candle::Device::Cpu,
            )?;
            Some((point_coords, point_labels))
        } else {
            None
        };
        let mask_inputs = if stage.metadata["has_mask_inputs"].as_bool().unwrap_or(false) {
            Some(load_tracker_fixture_tensor(
                bundle,
                stage.tensor_keys["mask_inputs"].as_str(),
            )?
            .to_dtype(DType::F32)?)
        } else {
            None
        };
        let high_res_features = if stage.tensor_keys.contains_key("high_res_features.0") {
            Some(vec![
                load_tracker_fixture_tensor(
                    bundle,
                    stage.tensor_keys["high_res_features.0"].as_str(),
                )?
                .to_dtype(DType::F32)?,
                load_tracker_fixture_tensor(
                    bundle,
                    stage.tensor_keys["high_res_features.1"].as_str(),
                )?
                .to_dtype(DType::F32)?,
            ])
        } else {
            None
        };
        let actual = model.forward_sam_heads(
            &backbone_features,
            point_prompt.as_ref(),
            mask_inputs.as_ref(),
            high_res_features.as_deref(),
            stage.metadata["multimask_output"].as_bool().unwrap_or(false),
            true,
        )?;
        let expected_low_res_masks = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["forward_sam_heads_output.low_res_masks"].as_str(),
        )?;
        let expected_high_res_masks = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["forward_sam_heads_output.high_res_masks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["forward_sam_heads_output.ious"].as_str(),
        )?;
        let expected_obj_ptr = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["forward_sam_heads_output.obj_ptr"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            bundle,
            stage.tensor_keys["forward_sam_heads_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "forward_sam_heads low_res_masks",
            &actual.low_res_masks,
            &expected_low_res_masks,
            low_res_mask_atol,
        )?;
        assert_tensor_close(
            "forward_sam_heads high_res_masks",
            &actual.high_res_masks,
            &expected_high_res_masks,
            high_res_mask_atol,
        )?;
        assert_tensor_close(
            "forward_sam_heads ious",
            &actual.iou_scores,
            &expected_ious,
            iou_atol,
        )?;
        assert_tensor_close(
            "forward_sam_heads obj_ptr",
            &actual.obj_ptr,
            &expected_obj_ptr,
            obj_ptr_atol,
        )?;
        assert_tensor_close(
            "forward_sam_heads object_score_logits",
            &actual.object_score_logits,
            &expected_object_score_logits,
            object_score_atol,
        )?;
        Ok(())
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
            config
                .predictor
                .refinement_detector_cond_frame_removal_window,
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
            vec![
                1,
                1,
                config.shapes.low_res_mask_size,
                config.shapes.low_res_mask_size
            ]
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
    fn fixture_backed_tracker_tensor_shapes_match_default_runtime_upstream_bundle() -> Result<()> {
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
    fn fixture_backed_point_prompt_runtime_bundle_matches_exported_shapes() -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::PointSingleClick)?;
        let config = Sam3TrackerConfig::build_tracker(false);
        let prompt_encoder = tracker_record(&manifest, 0, "sam_prompt_encoder")?;
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_inputs.points.0")?,
            vec![1, 1, 2]
        );
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_inputs.points.1")?,
            vec![1, 1]
        );
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_output.sparse_embeddings")?,
            vec![1, 2, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_output.dense_embeddings")?,
            vec![
                1,
                config.hidden_dim,
                config.image_embedding_size(),
                config.image_embedding_size()
            ]
        );

        let mask_decoder = tracker_record(&manifest, 0, "sam_mask_decoder")?;
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.low_res_multimasks")?,
            vec![1, 3, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.ious")?,
            vec![1, 3]
        );
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.sam_output_tokens")?,
            vec![1, 3, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.object_score_logits")?,
            vec![1, 1]
        );

        let forward_sam_heads = tracker_record(&manifest, 0, "forward_sam_heads")?;
        assert_eq!(
            fixture_shape(forward_sam_heads, "forward_sam_heads_output.low_res_masks")?,
            vec![1, 1, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_shape(forward_sam_heads, "forward_sam_heads_output.high_res_masks")?,
            vec![1, 1, config.image_size, config.image_size]
        );
        assert_eq!(
            fixture_shape(forward_sam_heads, "forward_sam_heads_output.obj_ptr")?,
            vec![1, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(
                forward_sam_heads,
                "forward_sam_heads_output.object_score_logits"
            )?,
            vec![1, 1]
        );

        let track_step = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(
            fixture_shape(track_step, "track_step_output.pred_masks")?,
            vec![1, 1, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_shape(track_step, "track_step_output.pred_masks_high_res")?,
            vec![1, 1, config.image_size, config.image_size]
        );
        Ok(())
    }

    #[test]
    fn tracker_track_frame_matches_single_click_point_fixture_values() -> Result<()> {
        assert_prompt_frame_point_fixture_matches(
            TrackerFixtureBundle::PointSingleClick,
            1,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_track_frame_matches_multi_click_point_fixture_values() -> Result<()> {
        assert_prompt_frame_point_fixture_matches(
            TrackerFixtureBundle::PointMultiClick,
            4,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_track_frame_matches_all_points_fixture_values() -> Result<()> {
        assert_prompt_frame_point_fixture_matches(
            TrackerFixtureBundle::PointAllPoints,
            6,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_track_frame_matches_mask_prompt_fixture_values() -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_checkpoint()? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MaskDirect)?;
        let use_mask_stage = tracker_record(&manifest, 0, "use_mask_as_output")?;
        let track_stage = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(track_stage.metadata["has_mask_inputs"].as_bool(), Some(true));
        let visual = build_fixture_visual_output(TrackerFixtureBundle::MaskDirect, use_mask_stage)?;
        let mask_input = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            track_stage.tensor_keys["mask_inputs"].as_str(),
        )?;
        let actual = model.track_frame(
            &visual,
            0,
            30,
            None,
            None,
            None,
            Some(&mask_input),
            &BTreeMap::new(),
            true,
            false,
            true,
            false,
        )?;
        let expected_low_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.low_res_masks"].as_str(),
        )?;
        let expected_high_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.high_res_masks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.ious"].as_str(),
        )?;
        let expected_obj_ptr = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.obj_ptr"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "mask prompt low_res_masks",
            &actual.state.low_res_masks,
            &expected_low_res_masks,
            5e-4,
        )?;
        assert_tensor_close(
            "mask prompt high_res_masks",
            &actual.state.high_res_masks,
            &expected_high_res_masks,
            1e-5,
        )?;
        assert_tensor_close(
            "mask prompt iou_scores",
            &actual.state.iou_scores,
            &expected_ious,
            1e-5,
        )?;
        assert_tensor_close(
            "mask prompt obj_ptr",
            &actual.state.obj_ptr,
            &expected_obj_ptr,
            0.5,
        )?;
        assert_tensor_close(
            "mask prompt object_score_logits",
            &actual.state.object_score_logits,
            &expected_object_score_logits,
            1e-5,
        )?;
        Ok(())
    }

    #[test]
    fn tracker_mask_decoder_matches_single_click_fixture_values() -> Result<()> {
        assert_mask_decoder_fixture_matches(
            TrackerFixtureBundle::PointSingleClick,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_mask_decoder_matches_multimask_disabled_sam_fixture_values() -> Result<()> {
        assert_mask_decoder_fixture_matches(
            TrackerFixtureBundle::MultimaskDisabledSam,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_forward_sam_heads_matches_single_click_fixture_values() -> Result<()> {
        assert_forward_sam_heads_fixture_matches(
            TrackerFixtureBundle::PointSingleClick,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_forward_sam_heads_matches_all_points_fixture_values() -> Result<()> {
        assert_forward_sam_heads_fixture_matches(
            TrackerFixtureBundle::PointAllPoints,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_forward_sam_heads_matches_multimask_disabled_sam_fixture_values() -> Result<()> {
        assert_forward_sam_heads_fixture_matches(
            TrackerFixtureBundle::MultimaskDisabledSam,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_use_mask_as_output_matches_direct_mask_fixture_values() -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::MaskDirect)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MaskDirect)?;
        let stage = tracker_record(&manifest, 0, "use_mask_as_output")?;
        let backbone_features = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["backbone_features"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let high_res_features = vec![
            load_tracker_fixture_tensor(
                TrackerFixtureBundle::MaskDirect,
                stage.tensor_keys["high_res_features.0"].as_str(),
            )?
            .to_dtype(DType::F32)?,
            load_tracker_fixture_tensor(
                TrackerFixtureBundle::MaskDirect,
                stage.tensor_keys["high_res_features.1"].as_str(),
            )?
            .to_dtype(DType::F32)?,
        ];
        let mask_inputs = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["mask_inputs"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let actual = model.use_mask_as_output(
            &backbone_features,
            Some(high_res_features.as_slice()),
            &mask_inputs,
            true,
        )?;
        let expected_low_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.low_res_masks"].as_str(),
        )?;
        let expected_high_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.high_res_masks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.ious"].as_str(),
        )?;
        let expected_obj_ptr = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.obj_ptr"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "use_mask_as_output low_res_masks",
            &actual.low_res_masks,
            &expected_low_res_masks,
            5e-4,
        )?;
        assert_tensor_close(
            "use_mask_as_output high_res_masks",
            &actual.high_res_masks,
            &expected_high_res_masks,
            1e-5,
        )?;
        assert_tensor_close(
            "use_mask_as_output ious",
            &actual.iou_scores,
            &expected_ious,
            1e-5,
        )?;
        assert_tensor_close(
            "use_mask_as_output obj_ptr",
            &actual.obj_ptr,
            &expected_obj_ptr,
            0.5,
        )?;
        assert_tensor_close(
            "use_mask_as_output object_score_logits",
            &actual.object_score_logits,
            &expected_object_score_logits,
            1e-5,
        )?;
        Ok(())
    }

    #[test]
    fn tracker_get_tpos_enc_matches_default_fixture_values() -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::Default)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::Default)?;
        let stage = tracker_record(&manifest, 1, "prepare_memory_conditioned_features")?;
        let offsets = stage.metadata["selected_object_pointer_temporal_offsets"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "default fixture missing selected_object_pointer_temporal_offsets".into(),
                )
            })?
            .iter()
            .map(|value| value.as_i64().unwrap_or_default())
            .collect::<Vec<_>>();
        let max_abs_pos = stage.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg("default fixture missing max_obj_ptrs_in_encoder".into())
            })? as usize;
        let expected = load_tracker_fixture_tensor(
            TrackerFixtureBundle::Default,
            stage.tensor_keys["object_pointer_temporal_pos_enc"].as_str(),
        )?;
        let actual =
            model.get_tpos_enc(offsets.as_slice(), &candle::Device::Cpu, Some(max_abs_pos), false)?;
        assert_tensor_close("get_tpos_enc", &actual, &expected, 1e-2)?;
        Ok(())
    }

    #[test]
    fn tracker_use_multimask_matches_fixture_branch_decisions() -> Result<()> {
        let Some(default_model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::PointSingleClick)?
        else {
            return Ok(());
        };
        let Some(disabled_tracking_model) = load_runtime_tracker_model_from_bundle(
            TrackerFixtureBundle::MultimaskDisabledTracking,
        )? else {
            return Ok(());
        };
        let Some(disabled_sam_model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::MultimaskDisabledSam)?
        else {
            return Ok(());
        };
        assert!(default_model.use_multimask(true, 1));
        assert!(!default_model.use_multimask(true, 4));
        assert!(!default_model.use_multimask(true, 6));
        assert!(disabled_tracking_model.use_multimask(true, 1));
        assert!(!disabled_tracking_model.use_multimask(false, 0));
        assert!(!disabled_sam_model.use_multimask(true, 1));
        Ok(())
    }

    #[test]
    fn default_box_bundle_routes_through_visual_prompt_before_tracker_runtime() -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::Default)?;
        let visual_prompt_stage = tracker_record(&manifest, 0, "get_visual_prompt")?;
        let prompt_stage = tracker_record(&manifest, 0, "sam_prompt_encoder")?;
        let forward_stage = tracker_record(&manifest, 0, "forward_sam_heads")?;
        assert_eq!(
            visual_prompt_stage.metadata["input_box_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            visual_prompt_stage.metadata["created_visual_prompt"].as_bool(),
            Some(true)
        );
        assert_eq!(prompt_stage.metadata["has_boxes"].as_bool(), Some(false));
        assert_eq!(
            forward_stage.metadata["has_point_inputs"].as_bool(),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn tracker_track_frame_executes_prompt_frame_point_path() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let point_coords = Tensor::from_vec(vec![12f32, 18f32], (1, 1, 2), &device)?;
        let point_labels = Tensor::from_vec(vec![1f32], (1, 1), &device)?;
        let output = model.track_frame(
            &dummy_visual(&device)?,
            0,
            1,
            Some(&point_coords),
            Some(&point_labels),
            None,
            None,
            &BTreeMap::new(),
            true,
            false,
            false,
            false,
        )?;
        assert_eq!(output.state.low_res_masks.dims4()?, (1, 1, 16, 16));
        assert_eq!(output.state.high_res_masks.dims4()?, (1, 1, 56, 56));
        assert_eq!(output.state.obj_ptr.dims2()?, (1, 32));
        assert_eq!(output.state.object_score_logits.dims2()?, (1, 1));
        assert!(output.state.maskmem_features.is_none());
        assert!(output.state.maskmem_pos_enc.is_none());
        assert_eq!(output.prompt_frame_indices, vec![0]);
        assert!(output.memory_frame_indices.is_empty());
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

    #[test]
    fn tracker_track_frame_still_reports_strict_port_status_for_memory_conditioning() -> Result<()>
    {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let point_coords = Tensor::from_vec(vec![12f32, 18f32], (1, 1, 2), &device)?;
        let point_labels = Tensor::from_vec(vec![1f32], (1, 1), &device)?;
        let mut history = BTreeMap::new();
        history.insert(0, dummy_state(&device)?);
        let err = model
            .track_frame(
                &dummy_visual(&device)?,
                1,
                2,
                Some(&point_coords),
                Some(&point_labels),
                None,
                None,
                &history,
                false,
                false,
                true,
                false,
            )
            .expect_err("memory-conditioned tracker path should remain blocked");
        assert!(err.to_string().contains("memory-conditioned tracking"));
        Ok(())
    }

    #[test]
    fn tracker_track_frame_executes_prompt_frame_mask_path() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let mask_input = Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?;
        let output = model.track_frame(
            &dummy_visual(&device)?,
            0,
            1,
            None,
            None,
            None,
            Some(&mask_input),
            &BTreeMap::new(),
            true,
            false,
            true,
            false,
        )?;
        assert_eq!(output.state.low_res_masks.dims4()?, (1, 1, 16, 16));
        assert_eq!(output.state.high_res_masks.dims4()?, (1, 1, 64, 64));
        assert_eq!(output.state.iou_scores.to_vec2::<f32>()?, vec![vec![1.0]]);
        assert!(output.prompt_frame_indices == vec![0]);
        assert!(output.memory_frame_indices.is_empty());
        Ok(())
    }
}
