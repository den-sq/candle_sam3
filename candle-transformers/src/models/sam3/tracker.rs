use std::collections::BTreeMap;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Embedding, LayerNorm, Module,
    VarBuilder,
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

#[derive(Debug)]
struct TrackerRoPEAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    freqs_real: Tensor,
    freqs_imag: Tensor,
    rope_k_repeat: bool,
}

impl TrackerRoPEAttention {
    fn new(config: &Sam3TrackerAttentionConfig, device: &Device) -> Result<Self> {
        if config.embedding_dim % config.num_heads != 0 {
            candle::bail!(
                "tracker attention embedding_dim {} must be divisible by num_heads {}",
                config.embedding_dim,
                config.num_heads
            );
        }
        let head_dim = config.embedding_dim / config.num_heads;
        if head_dim % 4 != 0 {
            candle::bail!("tracker attention head_dim must be divisible by 4, got {head_dim}");
        }
        let (freqs_real, freqs_imag) = compute_tracker_axial_freqs(
            head_dim,
            config.feat_sizes[0],
            config.feat_sizes[1],
            config.rope_theta,
            device,
        )?;
        Ok(Self {
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            freqs_real,
            freqs_imag,
            rope_k_repeat: config.rope_k_repeat,
        })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let in_dtype = q.dtype();
        let (batch_size, q_seq_len, q_dim) = q.dims3()?;
        let (_, k_seq_len, k_dim) = k.dims3()?;
        let (_, v_seq_len, v_dim) = v.dims3()?;
        if k_seq_len != v_seq_len {
            candle::bail!(
                "tracker attention expected key/value sequence lengths to match, got {k_seq_len} and {v_seq_len}"
            );
        }
        if q_dim != self.num_heads * self.head_dim
            || k_dim != self.num_heads * self.head_dim
            || v_dim != self.num_heads * self.head_dim
        {
            candle::bail!(
                "tracker attention expected q/k/v dims to match num_heads*head_dim={}, got q={}, k={}, v={}",
                self.num_heads * self.head_dim,
                q_dim,
                k_dim,
                v_dim
            );
        }
        let q = q
            .reshape((batch_size, q_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let k = k
            .reshape((batch_size, k_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let v = v
            .reshape((batch_size, v_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let (q, k) = apply_tracker_axial_rotary(
            &q,
            &k,
            &self.freqs_real,
            &self.freqs_imag,
            self.rope_k_repeat,
            num_k_exclude_rope,
        )?;
        let q = (q * self.scale)?;
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        out.transpose(1, 2)?
            .reshape((batch_size, q_seq_len, self.num_heads * self.head_dim))?
            .to_dtype(in_dtype)
    }
}

#[derive(Debug)]
struct TrackerTransformerLayer {
    self_attn_q_proj: Linear,
    self_attn_k_proj: Linear,
    self_attn_v_proj: Linear,
    self_attn_out_proj: Linear,
    cross_attn_q_proj: Linear,
    cross_attn_k_proj: Linear,
    cross_attn_v_proj: Linear,
    cross_attn_out_proj: Linear,
    self_attention_rope: TrackerRoPEAttention,
    cross_attention_rope: TrackerRoPEAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    pos_enc_at_attn: bool,
    pos_enc_at_cross_attn_queries: bool,
    pos_enc_at_cross_attn_keys: bool,
    cross_attention_first: bool,
}

impl TrackerTransformerLayer {
    fn new(config: &Sam3TrackerTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let cross_kv_dim = config.cross_attention.kv_in_dim.unwrap_or(d_model);
        let self_attn_q_proj = linear(vb.pp("self_attn").pp("q_proj"), d_model, d_model, true)?;
        let self_attn_k_proj = linear(vb.pp("self_attn").pp("k_proj"), d_model, d_model, true)?;
        let self_attn_v_proj = linear(vb.pp("self_attn").pp("v_proj"), d_model, d_model, true)?;
        let self_attn_out_proj = linear(vb.pp("self_attn").pp("out_proj"), d_model, d_model, true)?;
        let cross_attn_q_proj = linear(
            vb.pp("cross_attn_image").pp("q_proj"),
            d_model,
            d_model,
            true,
        )?;
        let cross_attn_k_proj = linear(
            vb.pp("cross_attn_image").pp("k_proj"),
            cross_kv_dim,
            d_model,
            true,
        )?;
        let cross_attn_v_proj = linear(
            vb.pp("cross_attn_image").pp("v_proj"),
            cross_kv_dim,
            d_model,
            true,
        )?;
        let cross_attn_out_proj = linear(
            vb.pp("cross_attn_image").pp("out_proj"),
            d_model,
            d_model,
            true,
        )?;
        Ok(Self {
            self_attn_q_proj,
            self_attn_k_proj,
            self_attn_v_proj,
            self_attn_out_proj,
            cross_attn_q_proj,
            cross_attn_k_proj,
            cross_attn_v_proj,
            cross_attn_out_proj,
            self_attention_rope: TrackerRoPEAttention::new(&config.self_attention, vb.device())?,
            cross_attention_rope: TrackerRoPEAttention::new(&config.cross_attention, vb.device())?,
            linear1: linear(
                vb.pp("linear1"),
                d_model,
                config.layer.dim_feedforward,
                true,
            )?,
            linear2: linear(
                vb.pp("linear2"),
                config.layer.dim_feedforward,
                d_model,
                true,
            )?,
            norm1: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
            norm3: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm3"))?,
            pos_enc_at_attn: config.layer.pos_enc_at_attn,
            pos_enc_at_cross_attn_queries: config.layer.pos_enc_at_cross_attn_queries,
            pos_enc_at_cross_attn_keys: config.layer.pos_enc_at_cross_attn_keys,
            cross_attention_first: config.layer.cross_attention_first,
        })
    }

    fn forward_self_attention(&self, tgt: &Tensor, query_pos: Option<&Tensor>) -> Result<Tensor> {
        let tgt2 = self.norm1.forward(tgt)?;
        let qk = match (self.pos_enc_at_attn, query_pos) {
            (true, Some(query_pos)) => tgt2.broadcast_add(query_pos)?,
            _ => tgt2.clone(),
        };
        let q = self.self_attn_q_proj.forward(&qk)?;
        let k = self.self_attn_k_proj.forward(&qk)?;
        let v = self.self_attn_v_proj.forward(&tgt2)?;
        let out = self.self_attention_rope.forward(&q, &k, &v, 0)?;
        let tgt2 = self.self_attn_out_proj.forward(&out)?;
        tgt.broadcast_add(&tgt2)
    }

    fn forward_cross_attention(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt2 = self.norm2.forward(tgt)?;
        let mut q_input = tgt2.clone();
        if self.pos_enc_at_cross_attn_queries {
            if let Some(query_pos) = query_pos {
                q_input = q_input.broadcast_add(query_pos)?;
            }
        }
        let q = self.cross_attn_q_proj.forward(&q_input)?;
        let mut k_input = memory.clone();
        if self.pos_enc_at_cross_attn_keys {
            if let Some(memory_pos) = memory_pos {
                k_input = k_input.broadcast_add(memory_pos)?;
            }
        }
        let k = self.cross_attn_k_proj.forward(&k_input)?;
        let v = self.cross_attn_v_proj.forward(memory)?;
        let out = self
            .cross_attention_rope
            .forward(&q, &k, &v, num_k_exclude_rope)?;
        let tgt2 = self.cross_attn_out_proj.forward(&out)?;
        tgt.broadcast_add(&tgt2)
    }

    fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt = if self.cross_attention_first {
            let tgt = self.forward_cross_attention(
                tgt,
                memory,
                query_pos,
                memory_pos,
                num_k_exclude_rope,
            )?;
            self.forward_self_attention(&tgt, query_pos)?
        } else {
            let tgt = self.forward_self_attention(tgt, query_pos)?;
            self.forward_cross_attention(&tgt, memory, query_pos, memory_pos, num_k_exclude_rope)?
        };
        let tgt2 = self.norm3.forward(&tgt)?;
        let tgt2 = self
            .linear2
            .forward(&self.linear1.forward(&tgt2)?.relu()?)?;
        tgt.broadcast_add(&tgt2)
    }
}

#[derive(Debug)]
struct TrackerMemoryTransformer {
    layers: Vec<TrackerTransformerLayer>,
    norm: LayerNorm,
    pos_enc_at_input: bool,
}

impl TrackerMemoryTransformer {
    fn new(config: &Sam3TrackerTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(config.encoder.num_layers);
        for layer_idx in 0..config.encoder.num_layers {
            layers.push(TrackerTransformerLayer::new(
                config,
                layers_vb.pp(layer_idx),
            )?);
        }
        Ok(Self {
            layers,
            norm: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm"))?,
            pos_enc_at_input: config.encoder.pos_enc_at_input,
        })
    }

    fn forward(
        &self,
        src: &Tensor,
        memory: &Tensor,
        src_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor> {
        let mut output = if self.pos_enc_at_input {
            match src_pos {
                Some(src_pos) => src.broadcast_add(&(src_pos * 0.1f64)?)?,
                None => src.clone(),
            }
        } else {
            src.clone()
        };
        for layer in self.layers.iter() {
            output = layer.forward(&output, memory, src_pos, memory_pos, num_obj_ptr_tokens)?;
        }
        self.norm.forward(&output)
    }
}

#[derive(Debug)]
struct TrackerSimpleMaskDownSampler {
    interpol_size: [usize; 2],
    convs: Vec<Conv2d>,
    norms: Vec<LayerNorm2d>,
    out_proj: Conv2d,
}

impl TrackerSimpleMaskDownSampler {
    fn new(
        config: &Sam3TrackerMaskDownsamplerConfig,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let total_stride = 16usize;
        let stride = config.stride;
        let mut num_layers = 0usize;
        let mut current_stride = 1usize;
        while current_stride < total_stride {
            current_stride *= stride;
            num_layers += 1;
        }
        if current_stride != total_stride {
            candle::bail!(
                "tracker simple mask downsampler expected total_stride {total_stride} to be divisible by stride {}, got effective stride {current_stride}",
                stride
            );
        }

        let encoder_vb = vb.pp("encoder");
        let mut convs = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers);
        let mut mask_in_chans = 1usize;
        let mut mask_out_chans = 1usize;
        for layer_idx in 0..num_layers {
            mask_out_chans *= stride * stride;
            convs.push(candle_nn::conv2d(
                mask_in_chans,
                mask_out_chans,
                config.kernel_size,
                Conv2dConfig {
                    stride,
                    padding: config.padding,
                    ..Default::default()
                },
                encoder_vb.pp(layer_idx * 3),
            )?);
            norms.push(LayerNorm2d::new(
                mask_out_chans,
                1e-6,
                encoder_vb.pp(layer_idx * 3 + 1),
            )?);
            mask_in_chans = mask_out_chans;
        }
        let out_proj = candle_nn::conv2d(
            mask_out_chans,
            embed_dim,
            1,
            Default::default(),
            encoder_vb.pp(num_layers * 3),
        )?;
        Ok(Self {
            interpol_size: config.interpol_size,
            convs,
            norms,
            out_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.to_dtype(DType::F32)?;
        let (_, _, height, width) = xs.dims4()?;
        if [height, width] != self.interpol_size {
            xs = resize_bilinear2d_antialias(&xs, self.interpol_size[0], self.interpol_size[1])?;
        }
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            xs = conv.forward(&xs)?;
            xs = norm.forward(&xs)?;
            xs = xs.gelu_erf()?;
        }
        self.out_proj.forward(&xs)
    }
}

#[derive(Debug)]
struct TrackerCxBlock {
    dwconv: Conv2d,
    norm: LayerNorm2d,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl TrackerCxBlock {
    fn new(config: &Sam3TrackerCxBlockConfig, vb: VarBuilder) -> Result<Self> {
        let groups = if config.use_dwconv { config.dim } else { 1 };
        let dwconv = candle_nn::conv2d(
            config.dim,
            config.dim,
            config.kernel_size,
            Conv2dConfig {
                padding: config.padding,
                groups,
                ..Default::default()
            },
            vb.pp("dwconv"),
        )?;
        let gamma = if config.layer_scale_init_value > 0.0 {
            Some(vb.get((config.dim,), "gamma")?)
        } else {
            None
        };
        Ok(Self {
            dwconv,
            norm: LayerNorm2d::new(config.dim, 1e-6, vb.pp("norm"))?,
            pwconv1: linear(vb.pp("pwconv1"), config.dim, 4 * config.dim, true)?,
            pwconv2: linear(vb.pp("pwconv2"), 4 * config.dim, config.dim, true)?,
            gamma,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.dwconv.forward(xs)?;
        xs = self.norm.forward(&xs)?;
        // CUDA linear falls back to a broadcasted 4D matmul for non-contiguous NHWC inputs,
        // which rejects the permuted stride layout produced here. Normalize to contiguous NHWC
        // before the pointwise projections so the fast flatten+matmul path is used instead.
        xs = xs.permute((0, 2, 3, 1))?.contiguous()?;
        xs = self.pwconv1.forward(&xs)?;
        xs = xs.gelu_erf()?.contiguous()?;
        xs = self.pwconv2.forward(&xs)?;
        if let Some(gamma) = self.gamma.as_ref() {
            xs = xs.broadcast_mul(&gamma.reshape((1, 1, 1, gamma.dim(0)?))?)?;
        }
        xs = xs.permute((0, 3, 1, 2))?;
        residual.broadcast_add(&xs)
    }
}

#[derive(Debug)]
struct TrackerSimpleFuser {
    layers: Vec<TrackerCxBlock>,
}

impl TrackerSimpleFuser {
    fn new(
        fuser_config: &Sam3TrackerFuserConfig,
        cx_block_config: &Sam3TrackerCxBlockConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(fuser_config.num_layers);
        for layer_idx in 0..fuser_config.num_layers {
            layers.push(TrackerCxBlock::new(
                cx_block_config,
                layers_vb.pp(layer_idx),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
struct TrackerSimpleMaskEncoder {
    mask_downsampler: TrackerSimpleMaskDownSampler,
    pix_feat_proj: Conv2d,
    fuser: TrackerSimpleFuser,
    out_proj: Conv2d,
    position_num_pos_feats: usize,
    position_normalize: bool,
    position_scale: f32,
    position_temperature: f32,
}

impl TrackerSimpleMaskEncoder {
    fn new(
        config: &Sam3TrackerMaskmemBackboneConfig,
        hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            mask_downsampler: TrackerSimpleMaskDownSampler::new(
                &config.mask_downsampler,
                hidden_dim,
                vb.pp("mask_downsampler"),
            )?,
            pix_feat_proj: candle_nn::conv2d(
                hidden_dim,
                hidden_dim,
                1,
                Default::default(),
                vb.pp("pix_feat_proj"),
            )?,
            fuser: TrackerSimpleFuser::new(&config.fuser, &config.cx_block, vb.pp("fuser"))?,
            out_proj: candle_nn::conv2d(
                hidden_dim,
                config.out_dim,
                1,
                Default::default(),
                vb.pp("out_proj"),
            )?,
            position_num_pos_feats: config.position_encoding.num_pos_feats,
            position_normalize: config.position_encoding.normalize,
            position_scale: config
                .position_encoding
                .scale
                .unwrap_or(2.0 * std::f32::consts::PI),
            position_temperature: config.position_encoding.temperature,
        })
    }

    fn forward(
        &self,
        pix_feat: &Tensor,
        masks: &Tensor,
        skip_mask_sigmoid: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut masks = if skip_mask_sigmoid {
            masks.clone()
        } else {
            candle_nn::ops::sigmoid(masks)?
        };
        masks = self.mask_downsampler.forward(&masks)?;
        let pix_feat = pix_feat.to_device(masks.device())?;
        let mut xs = self.pix_feat_proj.forward(&pix_feat)?;
        xs = xs.broadcast_add(&masks)?;
        xs = self.fuser.forward(&xs)?;
        xs = self.out_proj.forward(&xs)?;
        let pos = build_tracker_2d_sine_position_encoding(
            &xs,
            self.position_num_pos_feats,
            self.position_normalize,
            self.position_scale,
            self.position_temperature,
        )?
        .to_dtype(xs.dtype())?;
        Ok((xs, pos))
    }
}

fn compute_tracker_axial_freqs(
    dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if dim % 4 != 0 {
        candle::bail!("tracker rotary dim must be divisible by 4, got {dim}");
    }
    let rotary_dim = dim / 4;
    let seq_len = end_x * end_y;
    let inv_freqs: Vec<f32> = (0..rotary_dim)
        .map(|i| 1f32 / theta.powf((4 * i) as f32 / dim as f32))
        .collect();
    let mut freqs_real = vec![0f32; seq_len * (dim / 2)];
    let mut freqs_imag = vec![0f32; seq_len * (dim / 2)];
    for flat_idx in 0..seq_len {
        let x_pos = (flat_idx % end_x) as f32;
        let y_pos = (flat_idx / end_x) as f32;
        let row_real = &mut freqs_real[flat_idx * (dim / 2)..(flat_idx + 1) * (dim / 2)];
        let row_imag = &mut freqs_imag[flat_idx * (dim / 2)..(flat_idx + 1) * (dim / 2)];
        for (i, inv_freq) in inv_freqs.iter().copied().enumerate() {
            let x_freq = x_pos * inv_freq;
            let y_freq = y_pos * inv_freq;
            row_real[i] = x_freq.cos();
            row_imag[i] = x_freq.sin();
            row_real[rotary_dim + i] = y_freq.cos();
            row_imag[rotary_dim + i] = y_freq.sin();
        }
    }
    Ok((
        Tensor::from_slice(&freqs_real, (seq_len, dim / 2), device)?,
        Tensor::from_slice(&freqs_imag, (seq_len, dim / 2), device)?,
    ))
}

fn apply_tracker_rotary_single(
    xs: &Tensor,
    freqs_real: &Tensor,
    freqs_imag: &Tensor,
) -> Result<Tensor> {
    let (batch_size, num_heads, seq_len, head_dim) = xs.dims4()?;
    let xs_dtype = xs.dtype();
    let xs = xs
        .to_dtype(DType::F32)?
        .reshape((batch_size, num_heads, seq_len, head_dim / 2, 2))?;
    let xs_real = xs.i((.., .., .., .., 0))?;
    let xs_imag = xs.i((.., .., .., .., 1))?;
    let freqs_real = freqs_real.reshape((1, 1, seq_len, head_dim / 2))?;
    let freqs_imag = freqs_imag.reshape((1, 1, seq_len, head_dim / 2))?;
    let real = (xs_real.broadcast_mul(&freqs_real)? - xs_imag.broadcast_mul(&freqs_imag)?)?;
    let imag = (xs_real.broadcast_mul(&freqs_imag)? + xs_imag.broadcast_mul(&freqs_real)?)?;
    Tensor::stack(&[&real, &imag], 4)?
        .reshape((batch_size, num_heads, seq_len, head_dim))?
        .to_dtype(xs_dtype)
}

fn apply_tracker_axial_rotary(
    q: &Tensor,
    k: &Tensor,
    freqs_real: &Tensor,
    freqs_imag: &Tensor,
    repeat_freqs_k: bool,
    num_k_exclude_rope: usize,
) -> Result<(Tensor, Tensor)> {
    let q_seq_len = q.dim(2)?;
    let k_seq_len = k.dim(2)?;
    let q_freqs_real = freqs_real.narrow(0, 0, q_seq_len)?;
    let q_freqs_imag = freqs_imag.narrow(0, 0, q_seq_len)?;
    let q = apply_tracker_rotary_single(q, &q_freqs_real, &q_freqs_imag)?;

    let num_k_rope = k_seq_len.saturating_sub(num_k_exclude_rope);
    if num_k_rope == 0 {
        return Ok((q, k.clone()));
    }
    let mut k_freqs_real = q_freqs_real.clone();
    let mut k_freqs_imag = q_freqs_imag.clone();
    if num_k_rope != q_seq_len {
        if !repeat_freqs_k || num_k_rope % q_seq_len != 0 {
            candle::bail!(
                "tracker rotary expected key rope length {num_k_rope} to equal query length {q_seq_len} or be a whole-number repeat"
            );
        }
        let repeat_factor = num_k_rope / q_seq_len;
        k_freqs_real = k_freqs_real.repeat((repeat_factor, 1))?;
        k_freqs_imag = k_freqs_imag.repeat((repeat_factor, 1))?;
    }
    let k_rope = k.narrow(2, 0, num_k_rope)?;
    let k_rope = apply_tracker_rotary_single(&k_rope, &k_freqs_real, &k_freqs_imag)?;
    let k = if num_k_rope < k_seq_len {
        let k_tail = k.narrow(2, num_k_rope, k_seq_len - num_k_rope)?;
        Tensor::cat(&[&k_rope, &k_tail], 2)?
    } else {
        k_rope
    };
    Ok((q, k))
}

#[derive(Debug)]
struct PreparedMemoryConditioning {
    pix_feat_with_mem: Tensor,
    selected_conditioning_frame_indices: Vec<usize>,
    selected_memory_frame_indices: Vec<usize>,
    selected_object_pointer_frame_indices: Vec<usize>,
}

#[derive(Debug)]
struct PreparedMemoryPrompt {
    prompt: Option<Tensor>,
    prompt_pos: Option<Tensor>,
    num_obj_ptr_tokens: usize,
    selected_conditioning_frame_indices: Vec<usize>,
    selected_memory_frame_indices: Vec<usize>,
    selected_object_pointer_frame_indices: Vec<usize>,
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
                    let upscaled_embedding =
                        self.output_upscaling_conv2.forward(&upscaled_embedding)?;
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
    maskmem_backbone: TrackerSimpleMaskEncoder,
    memory_transformer: TrackerMemoryTransformer,
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
        let maskmem_backbone = TrackerSimpleMaskEncoder::new(
            &config.maskmem_backbone,
            config.hidden_dim,
            vb.pp("maskmem_backbone"),
        )?;
        let memory_transformer =
            TrackerMemoryTransformer::new(&config.transformer, vb.pp("transformer").pp("encoder"))?;
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
            maskmem_backbone,
            memory_transformer,
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
            return Tensor::zeros(
                (rel_pos_list.len(), self.config.memory_dim),
                DType::F32,
                device,
            );
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

    fn cal_mem_score(&self, object_score_logits: &Tensor, iou_score: &Tensor) -> Result<f32> {
        let zeros = Tensor::zeros_like(object_score_logits)?;
        let object_score_norm = object_score_logits.gt(0f64)?.where_cond(
            &((candle_nn::ops::sigmoid(object_score_logits)? * 2f64)? - 1f64)?,
            &zeros,
        )?;
        object_score_norm
            .broadcast_mul(iou_score)?
            .mean_all()?
            .to_dtype(DType::F32)?
            .to_vec0::<f32>()
    }

    fn frame_filter(
        &self,
        history: &BTreeMap<usize, TrackerFrameState>,
        track_in_reverse: bool,
        frame_idx: usize,
        num_frames: usize,
        r: usize,
    ) -> Result<Vec<usize>> {
        if (frame_idx == 0 && !track_in_reverse)
            || (frame_idx + 1 == num_frames && track_in_reverse)
        {
            return Ok(Vec::new());
        }

        let max_num = self.config.max_obj_ptrs_in_encoder.min(num_frames);
        let (start, end, step, must_include) = if !track_in_reverse {
            (
                frame_idx.saturating_sub(1) as isize,
                0isize,
                -(r as isize),
                frame_idx.saturating_sub(1),
            )
        } else {
            (
                (frame_idx + 1) as isize,
                num_frames as isize,
                r as isize,
                frame_idx + 1,
            )
        };

        let mut valid_indices = Vec::new();
        let mut i = start;
        while if !track_in_reverse { i >= end } else { i < end } {
            let frame = i as usize;
            let Some(state) = history.get(&frame) else {
                i += step;
                continue;
            };
            if state.is_cond_frame {
                i += step;
                continue;
            }
            let iou_score = state.iou_scores.max(D::Minus1)?;
            let score_per_frame = self.cal_mem_score(&state.object_score_logits, &iou_score)?;
            if score_per_frame > self.config.mf_threshold {
                valid_indices.insert(0, frame);
            }
            if valid_indices.len() >= max_num.saturating_sub(1) {
                break;
            }
            i += step;
        }
        if !valid_indices.contains(&must_include) {
            valid_indices.push(must_include);
        }
        Ok(valid_indices)
    }

    fn select_closest_cond_frame_indices(
        &self,
        frame_idx: usize,
        cond_frame_outputs: &BTreeMap<usize, &TrackerFrameState>,
    ) -> (Vec<usize>, Vec<usize>) {
        if self.config.max_cond_frames_in_attn == usize::MAX {
            return (cond_frame_outputs.keys().copied().collect(), Vec::new());
        }

        let mut selected = Vec::new();
        let push_unique = |items: &mut Vec<usize>, value: Option<usize>| {
            if let Some(value) = value {
                if !items.contains(&value) {
                    items.push(value);
                }
            }
        };
        if self.config.keep_first_cond_frame {
            let idx_first = cond_frame_outputs
                .keys()
                .copied()
                .filter(|t| *t < frame_idx)
                .min()
                .or_else(|| {
                    cond_frame_outputs
                        .keys()
                        .copied()
                        .filter(|t| *t > frame_idx)
                        .max()
                });
            push_unique(&mut selected, idx_first);
        }
        let idx_before = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| *t < frame_idx)
            .max();
        push_unique(&mut selected, idx_before);
        let idx_after = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| *t >= frame_idx)
            .min();
        push_unique(&mut selected, idx_after);

        let num_remain = self
            .config
            .max_cond_frames_in_attn
            .saturating_sub(selected.len());
        let mut remaining = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| !selected.contains(t))
            .collect::<Vec<_>>();
        remaining.sort_by_key(|t| {
            let abs = if *t >= frame_idx {
                *t - frame_idx
            } else {
                frame_idx - *t
            };
            (abs, *t)
        });
        selected.extend(remaining.into_iter().take(num_remain));

        let selected_set = selected.clone();
        let unselected = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| !selected_set.contains(t))
            .collect::<Vec<_>>();
        (selected, unselected)
    }

    fn prepare_memory_conditioned_features(
        &self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &[Tensor],
        current_vision_pos_embeds: &[Tensor],
        feat_sizes: &[(usize, usize)],
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        use_prev_mem_frame: bool,
    ) -> Result<PreparedMemoryConditioning> {
        let batch_size = current_vision_feats
            .last()
            .ok_or_else(|| {
                candle::Error::Msg("tracker requires at least one current vision feature".into())
            })?
            .dim(1)?;
        let channels = self.config.hidden_dim;
        let (height, width) = *feat_sizes.last().ok_or_else(|| {
            candle::Error::Msg("tracker requires at least one feature size".into())
        })?;
        if self.config.num_maskmem == 0 || is_init_cond_frame || !use_prev_mem_frame {
            let pix_feat_with_mem = current_vision_feats
                .last()
                .expect("checked above")
                .broadcast_add(&self.no_mem_embed)?
                .permute((1, 2, 0))?
                .reshape((batch_size, channels, height, width))?;
            return Ok(PreparedMemoryConditioning {
                pix_feat_with_mem,
                selected_conditioning_frame_indices: Vec::new(),
                selected_memory_frame_indices: Vec::new(),
                selected_object_pointer_frame_indices: Vec::new(),
            });
        }

        let cond_frame_outputs = history
            .iter()
            .filter_map(|(frame, state)| state.is_cond_frame.then_some((*frame, state)))
            .collect::<BTreeMap<_, _>>();
        if cond_frame_outputs.is_empty() {
            candle::bail!("tracker memory conditioning expected at least one conditioning frame");
        }
        let prepared_prompt = self.build_memory_conditioning_prompt(
            frame_idx,
            history,
            num_frames,
            track_in_reverse,
            &cond_frame_outputs,
        )?;
        let selected_conditioning_frame_indices =
            prepared_prompt.selected_conditioning_frame_indices.clone();
        let selected_object_pointer_frame_indices = prepared_prompt
            .selected_object_pointer_frame_indices
            .clone();
        let selected_memory_frame_indices = prepared_prompt.selected_memory_frame_indices.clone();

        let Some(prompt) = prepared_prompt.prompt else {
            let pix_feat_with_mem = current_vision_feats
                .last()
                .expect("checked above")
                .broadcast_add(&self.no_mem_embed)?
                .permute((1, 2, 0))?
                .reshape((batch_size, channels, height, width))?;
            return Ok(PreparedMemoryConditioning {
                pix_feat_with_mem,
                selected_conditioning_frame_indices,
                selected_memory_frame_indices,
                selected_object_pointer_frame_indices,
            });
        };
        let prompt_pos = prepared_prompt
            .prompt_pos
            .expect("prompt position encoding must exist whenever prompt exists");
        let num_obj_ptr_tokens = prepared_prompt.num_obj_ptr_tokens;
        let src = current_vision_feats
            .last()
            .expect("checked above")
            .transpose(0, 1)?
            .contiguous()?;
        let src_pos = current_vision_pos_embeds
            .last()
            .expect("checked above")
            .transpose(0, 1)?
            .contiguous()?;
        let prompt = prompt.transpose(0, 1)?.contiguous()?;
        let prompt_pos = prompt_pos.transpose(0, 1)?.contiguous()?;
        let encoded = self.memory_transformer.forward(
            &src,
            &prompt,
            Some(&src_pos),
            Some(&prompt_pos),
            num_obj_ptr_tokens,
        )?;
        let pix_feat_with_mem = encoded
            .transpose(1, 2)?
            .reshape((batch_size, channels, height, width))?;
        Ok(PreparedMemoryConditioning {
            pix_feat_with_mem,
            selected_conditioning_frame_indices,
            selected_memory_frame_indices,
            selected_object_pointer_frame_indices,
        })
    }

    fn build_memory_conditioning_prompt(
        &self,
        frame_idx: usize,
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        cond_frame_outputs: &BTreeMap<usize, &TrackerFrameState>,
    ) -> Result<PreparedMemoryPrompt> {
        let device = cond_frame_outputs
            .values()
            .next()
            .map(|state| state.obj_ptr.device())
            .ok_or_else(|| {
                candle::Error::Msg(
                    "tracker memory conditioning expected at least one conditioning frame".into(),
                )
            })?;
        let batch_size = cond_frame_outputs
            .values()
            .next()
            .map(|state| state.obj_ptr.dim(0))
            .transpose()?
            .ok_or_else(|| {
                candle::Error::Msg(
                    "tracker memory conditioning expected at least one conditioning frame".into(),
                )
            })?;
        let channels = self.config.hidden_dim;
        let (selected_cond_ordered, unselected_cond_indices) =
            self.select_closest_cond_frame_indices(frame_idx, cond_frame_outputs);
        let mut selected_conditioning_frame_indices = selected_cond_ordered.clone();
        selected_conditioning_frame_indices.sort_unstable();
        let unselected_cond_outputs = unselected_cond_indices
            .iter()
            .filter_map(|frame| cond_frame_outputs.get(frame).map(|state| (*frame, *state)))
            .collect::<BTreeMap<_, _>>();

        let tpos_sign_mul: i64 = if track_in_reverse { -1 } else { 1 };
        let mut prompt_parts = Vec::new();
        let mut prompt_pos_parts = Vec::new();
        let mut selected_memory_frame_indices_ordered = Vec::new();
        let mut selected_object_pointer_frame_indices = Vec::new();

        for &selected_frame in selected_cond_ordered.iter() {
            let prev = cond_frame_outputs
                .get(&selected_frame)
                .expect("selected conditioning frame missing from history");
            let Some(maskmem_features) = &prev.maskmem_features else {
                candle::bail!(
                    "conditioning frame {selected_frame} is missing maskmem_features required for tracker memory conditioning"
                );
            };
            let Some(maskmem_pos_enc) = &prev.maskmem_pos_enc else {
                candle::bail!(
                    "conditioning frame {selected_frame} is missing maskmem_pos_enc required for tracker memory conditioning"
                );
            };
            let maskmem_features = maskmem_features
                .to_device(device)?
                .to_dtype(self.no_obj_ptr.dtype())?;
            let maskmem_pos_enc = maskmem_pos_enc
                .to_device(device)?
                .to_dtype(self.no_obj_ptr.dtype())?;
            prompt_parts.push(maskmem_features.flatten(2, 3)?.permute((2, 0, 1))?);
            let pos = maskmem_pos_enc.flatten(2, 3)?.permute((2, 0, 1))?;
            let pos = pos.broadcast_add(&self.maskmem_tpos_enc.i(self.config.num_maskmem - 1)?)?;
            prompt_pos_parts.push(pos);
        }

        let r = self.config.memory_temporal_stride_for_eval.max(1);
        let valid_indices = if self.config.use_memory_selection {
            Some(self.frame_filter(history, track_in_reverse, frame_idx, num_frames, r)?)
        } else {
            None
        };
        for t_pos in 1..self.config.num_maskmem {
            let t_rel = self.config.num_maskmem - t_pos;
            let prev_frame_idx = if let Some(valid_indices) = valid_indices.as_ref() {
                if t_rel > valid_indices.len() {
                    continue;
                }
                valid_indices[valid_indices.len() - t_rel]
            } else if t_rel == 1 {
                if !track_in_reverse {
                    frame_idx.saturating_sub(t_rel)
                } else {
                    frame_idx + t_rel
                }
            } else if !track_in_reverse {
                let nearest = ((frame_idx.saturating_sub(2)) / r) * r;
                nearest.saturating_sub((t_rel - 2) * r)
            } else {
                let nearest = (frame_idx + 1).div_ceil(r) * r;
                nearest + (t_rel - 2) * r
            };
            let prev = history
                .get(&prev_frame_idx)
                .filter(|state| !state.is_cond_frame)
                .or_else(|| unselected_cond_outputs.get(&prev_frame_idx).copied());
            let Some(prev) = prev else {
                continue;
            };
            if prev.maskmem_features.is_none() || prev.maskmem_pos_enc.is_none() {
                continue;
            }
            let Some(maskmem_features) = &prev.maskmem_features else {
                candle::bail!(
                    "memory frame {prev_frame_idx} is missing maskmem_features required for tracker memory conditioning"
                );
            };
            let Some(maskmem_pos_enc) = &prev.maskmem_pos_enc else {
                candle::bail!(
                    "memory frame {prev_frame_idx} is missing maskmem_pos_enc required for tracker memory conditioning"
                );
            };
            let maskmem_features = maskmem_features
                .to_device(device)?
                .to_dtype(self.no_obj_ptr.dtype())?;
            let maskmem_pos_enc = maskmem_pos_enc
                .to_device(device)?
                .to_dtype(self.no_obj_ptr.dtype())?;
            prompt_parts.push(maskmem_features.flatten(2, 3)?.permute((2, 0, 1))?);
            let pos = maskmem_pos_enc.flatten(2, 3)?.permute((2, 0, 1))?;
            let pos = pos.broadcast_add(
                &self
                    .maskmem_tpos_enc
                    .i(self.config.num_maskmem - t_pos - 1)?,
            )?;
            prompt_pos_parts.push(pos);
            selected_memory_frame_indices_ordered.push(prev_frame_idx);
        }

        let max_obj_ptrs_in_encoder = self.config.max_obj_ptrs_in_encoder.min(num_frames);
        let ptr_cond_frames = if !track_in_reverse {
            selected_cond_ordered
                .iter()
                .copied()
                .filter(|t| *t <= frame_idx)
                .collect::<Vec<_>>()
        } else {
            selected_cond_ordered
                .iter()
                .copied()
                .filter(|t| *t >= frame_idx)
                .collect::<Vec<_>>()
        };
        let mut obj_ptr_tensors = Vec::new();
        let mut obj_ptr_offsets = Vec::new();
        for selected_frame in ptr_cond_frames.iter().copied() {
            selected_object_pointer_frame_indices.push(selected_frame);
            obj_ptr_offsets
                .push(((frame_idx as i64 - selected_frame as i64) * tpos_sign_mul) as i64);
            obj_ptr_tensors.push(
                cond_frame_outputs
                    .get(&selected_frame)
                    .expect("conditioning frame missing obj_ptr source")
                    .obj_ptr
                    .clone(),
            );
        }
        for t_diff in 1..max_obj_ptrs_in_encoder {
            let frame = if let Some(valid_indices) = valid_indices.as_ref() {
                if t_diff > valid_indices.len().saturating_sub(1) {
                    break;
                }
                valid_indices[valid_indices.len() - t_diff]
            } else if !track_in_reverse {
                let frame = frame_idx.saturating_sub(t_diff);
                if frame_idx < t_diff {
                    break;
                }
                frame
            } else {
                let frame = frame_idx + t_diff;
                if frame >= num_frames {
                    break;
                }
                frame
            };
            let prev = history
                .get(&frame)
                .filter(|state| !state.is_cond_frame)
                .or_else(|| unselected_cond_outputs.get(&frame).copied());
            if let Some(prev) = prev {
                selected_object_pointer_frame_indices.push(frame);
                obj_ptr_offsets.push(t_diff as i64);
                obj_ptr_tensors.push(prev.obj_ptr.clone());
            }
        }

        let mut num_obj_ptr_tokens = 0usize;
        if !obj_ptr_tensors.is_empty() {
            let mut obj_ptrs = Tensor::stack(obj_ptr_tensors.as_slice(), 0)?;
            let mut obj_pos = self.get_tpos_enc(
                obj_ptr_offsets.as_slice(),
                device,
                Some(max_obj_ptrs_in_encoder),
                false,
            )?;
            obj_pos = obj_pos.unsqueeze(1)?.expand((
                obj_ptr_offsets.len(),
                batch_size,
                self.config.memory_dim,
            ))?;
            if self.config.memory_dim < channels {
                let split = channels / self.config.memory_dim;
                obj_ptrs = obj_ptrs
                    .reshape((obj_ptrs.dim(0)?, batch_size, split, self.config.memory_dim))?
                    .permute((0, 2, 1, 3))?
                    .flatten(0, 1)?;
                obj_pos = repeat_interleave(&obj_pos, split, 0)?;
            }
            num_obj_ptr_tokens = obj_ptrs.dim(0)?;
            prompt_parts.push(obj_ptrs);
            prompt_pos_parts.push(obj_pos);
        }

        let mut selected_memory_frame_indices = selected_memory_frame_indices_ordered;
        selected_memory_frame_indices.sort_unstable();
        if prompt_parts.is_empty() {
            return Ok(PreparedMemoryPrompt {
                prompt: None,
                prompt_pos: None,
                num_obj_ptr_tokens,
                selected_conditioning_frame_indices,
                selected_memory_frame_indices,
                selected_object_pointer_frame_indices,
            });
        }

        let prompt_refs = prompt_parts.iter().collect::<Vec<_>>();
        let prompt_pos_refs = prompt_pos_parts.iter().collect::<Vec<_>>();
        Ok(PreparedMemoryPrompt {
            prompt: Some(Tensor::cat(prompt_refs.as_slice(), 0)?),
            prompt_pos: Some(Tensor::cat(prompt_pos_refs.as_slice(), 0)?),
            num_obj_ptr_tokens,
            selected_conditioning_frame_indices,
            selected_memory_frame_indices,
            selected_object_pointer_frame_indices,
        })
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
                candle::bail!(
                    "sam3 tracker image encoder expects CHW or BCHW input, got rank {rank}"
                )
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
        use_prev_mem_frame: bool,
        run_mem_encoder: bool,
    ) -> Result<TrackerStepOutput> {
        if _visual.backbone_fpn.is_empty() {
            candle::bail!("tracker requires at least one visual feature level")
        }
        if _visual.vision_pos_enc.is_empty() {
            candle::bail!("tracker requires at least one visual position-encoding level")
        }
        let compute_dtype = self.no_obj_ptr.dtype();
        let prepared_high_res_features = if _visual.backbone_fpn.len() > 1 {
            Some(self.prepare_high_res_features(
                &_visual.backbone_fpn[.._visual.backbone_fpn.len() - 1],
            )?)
        } else {
            None
        };
        let high_res_features = prepared_high_res_features.as_deref();
        let mut prompt_frame_indices = if is_conditioning_frame {
            vec![_frame_idx]
        } else {
            Vec::new()
        };
        let mut memory_frame_indices = Vec::new();
        let backbone_features = if !_history.is_empty() {
            let feat_sizes = _visual
                .backbone_fpn
                .iter()
                .zip(_visual.vision_pos_enc.iter())
                .map(|(feat, pos)| {
                    let (_, feat_channels, feat_h, feat_w) = feat.dims4()?;
                    let pos_shape = pos.dims4()?;
                    if pos_shape != (1, feat_channels, feat_h, feat_w) {
                        candle::bail!(
                            "tracker expected matching feature/pos shapes, got ({feat_channels}, {feat_h}, {feat_w}) and {pos_shape:?}"
                        );
                    }
                    Ok((feat_h, feat_w))
                })
                .collect::<Result<Vec<_>>>()?;
            let current_vision_feats = _visual
                .backbone_fpn
                .iter()
                .map(|feat| {
                    feat.to_dtype(compute_dtype)?
                        .permute((2, 3, 0, 1))?
                        .reshape((feat.dim(2)? * feat.dim(3)?, feat.dim(0)?, feat.dim(1)?))
                })
                .collect::<Result<Vec<_>>>()?;
            let current_vision_pos_embeds = _visual
                .vision_pos_enc
                .iter()
                .map(|pos| {
                    pos.to_dtype(compute_dtype)?
                        .permute((2, 3, 0, 1))?
                        .reshape((pos.dim(2)? * pos.dim(3)?, pos.dim(0)?, pos.dim(1)?))
                })
                .collect::<Result<Vec<_>>>()?;
            let prepared = self.prepare_memory_conditioned_features(
                _frame_idx,
                is_conditioning_frame,
                current_vision_feats.as_slice(),
                current_vision_pos_embeds.as_slice(),
                feat_sizes.as_slice(),
                _history,
                _num_frames,
                reverse,
                use_prev_mem_frame,
            )?;
            prompt_frame_indices = prepared.selected_conditioning_frame_indices;
            memory_frame_indices = prepared.selected_memory_frame_indices;
            prepared.pix_feat_with_mem.to_dtype(compute_dtype)?
        } else {
            _visual
                .backbone_fpn
                .last()
                .expect("checked non-empty above")
                .to_dtype(compute_dtype)?
        };
        if let Some(mask_input) = _mask_input {
            let mut state = self.use_mask_as_output(
                &backbone_features,
                high_res_features,
                mask_input,
                is_conditioning_frame,
            )?;
            if run_mem_encoder && self.config.num_maskmem > 0 {
                let (maskmem_features, maskmem_pos_enc) = self.encode_new_memory_from_visual(
                    _visual,
                    &state.high_res_masks,
                    &state.object_score_logits,
                    false,
                )?;
                state.maskmem_features = Some(maskmem_features);
                state.maskmem_pos_enc = Some(maskmem_pos_enc);
                state = self.maybe_offload_state_for_eval(state)?;
            }
            return Ok(TrackerStepOutput {
                state,
                prompt_frame_indices,
                memory_frame_indices,
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
        let mut state = self.forward_sam_heads(
            &backbone_features,
            point_prompt.as_ref(),
            None,
            high_res_features,
            multimask_output,
            is_conditioning_frame,
        )?;
        if run_mem_encoder && self.config.num_maskmem > 0 {
            let (maskmem_features, maskmem_pos_enc) = self.encode_new_memory_from_visual(
                _visual,
                &state.high_res_masks,
                &state.object_score_logits,
                point_prompt.is_some(),
            )?;
            state.maskmem_features = Some(maskmem_features);
            state.maskmem_pos_enc = Some(maskmem_pos_enc);
            state = self.maybe_offload_state_for_eval(state)?;
        }
        Ok(TrackerStepOutput {
            state,
            prompt_frame_indices,
            memory_frame_indices,
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
            let conv_s0 = self.sam_mask_decoder.conv_s0.as_ref().ok_or_else(|| {
                candle::Error::Msg("tracker high-res projection conv_s0 missing".into())
            })?;
            let conv_s1 = self.sam_mask_decoder.conv_s1.as_ref().ok_or_else(|| {
                candle::Error::Msg("tracker high-res projection conv_s1 missing".into())
            })?;
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

    fn apply_non_overlapping_constraints(&self, pred_masks: &Tensor) -> Result<Tensor> {
        let (batch_size, channels, height, width) = pred_masks.dims4()?;
        if batch_size == 1 {
            return Ok(pred_masks.clone());
        }
        let device = pred_masks.device();
        let max_obj_inds = pred_masks.argmax_keepdim(0)?;
        let batch_obj_inds =
            Tensor::arange(0u32, batch_size as u32, device)?.reshape((batch_size, 1, 1, 1))?;
        let keep = batch_obj_inds
            .broadcast_eq(&max_obj_inds.broadcast_as((batch_size, channels, height, width))?)?;
        let neg_ten = Tensor::full(-10f32, pred_masks.shape(), device)?;
        let suppressed = pred_masks.le(-10f64)?.where_cond(pred_masks, &neg_ten)?;
        keep.where_cond(pred_masks, &suppressed)
    }

    fn encode_new_memory_from_pix_feat(
        &self,
        pix_feat: &Tensor,
        pred_masks_high_res: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_pts: bool,
    ) -> Result<(Tensor, Tensor)> {
        let pix_feat = pix_feat.to_dtype(self.no_obj_ptr.dtype())?;
        let mut pred_masks_high_res =
            normalize_mask_prompt(pred_masks_high_res, pix_feat.device())?;
        let object_score_logits = object_score_logits
            .to_device(pix_feat.device())?
            .to_dtype(self.no_obj_ptr.dtype())?;
        if self.config.non_overlap_masks_for_mem_enc {
            pred_masks_high_res = self.apply_non_overlapping_constraints(&pred_masks_high_res)?;
        }
        let mut mask_for_mem = if is_mask_from_pts {
            pred_masks_high_res.gt(0f64)?.to_dtype(DType::F32)?
        } else {
            candle_nn::ops::sigmoid(&pred_masks_high_res)?
        };
        if self.config.sigmoid_scale_for_mem_enc != 1.0 {
            mask_for_mem =
                mask_for_mem.affine(self.config.sigmoid_scale_for_mem_enc as f64, 0.0)?;
        }
        if self.config.sigmoid_bias_for_mem_enc != 0.0 {
            mask_for_mem = mask_for_mem.affine(1.0, self.config.sigmoid_bias_for_mem_enc as f64)?;
        }
        let (mut maskmem_features, maskmem_pos_enc) =
            self.maskmem_backbone
                .forward(&pix_feat, &mask_for_mem, true)?;
        let appearing = object_score_logits
            .gt(0f64)?
            .to_dtype(maskmem_features.dtype())?
            .affine(-1.0, 1.0)?;
        let no_obj_embed = self
            .no_obj_embed_spatial
            .to_device(maskmem_features.device())?
            .to_dtype(maskmem_features.dtype())?
            .reshape((1, self.config.memory_dim, 1, 1))?;
        let no_obj_add = appearing
            .reshape((appearing.dim(0)?, 1, 1, 1))?
            .broadcast_mul(&no_obj_embed.broadcast_as(maskmem_features.shape())?)?;
        maskmem_features = maskmem_features.broadcast_add(&no_obj_add)?;
        Ok((maskmem_features, maskmem_pos_enc))
    }

    fn encode_new_memory_from_visual(
        &self,
        visual: &VisualBackboneOutput,
        pred_masks_high_res: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_pts: bool,
    ) -> Result<(Tensor, Tensor)> {
        let pix_feat = visual.backbone_fpn.last().ok_or_else(|| {
            candle::Error::Msg(
                "tracker memory encoder requires a top-level backbone feature".into(),
            )
        })?;
        self.encode_new_memory_from_pix_feat(
            pix_feat,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
        )
    }

    fn maybe_offload_state_for_eval(&self, state: TrackerFrameState) -> Result<TrackerFrameState> {
        if !self.config.predictor.offload_output_to_cpu_for_eval {
            return Ok(state);
        }
        let storage = &Device::Cpu;
        Ok(TrackerFrameState {
            low_res_masks: state.low_res_masks.to_device(storage)?,
            high_res_masks: state.high_res_masks.to_device(storage)?,
            iou_scores: state.iou_scores.to_device(storage)?,
            obj_ptr: state.obj_ptr,
            object_score_logits: state.object_score_logits,
            maskmem_features: state
                .maskmem_features
                .as_ref()
                .map(|tensor| tensor.to_dtype(DType::BF16)?.to_device(storage))
                .transpose()?,
            maskmem_pos_enc: state
                .maskmem_pos_enc
                .as_ref()
                .map(|tensor| tensor.to_device(storage))
                .transpose()?,
            is_cond_frame: state.is_cond_frame,
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
        visual: &VisualBackboneOutput,
        state: &TrackerFrameState,
    ) -> Result<(Tensor, Tensor)> {
        self.encode_new_memory_from_visual(
            visual,
            &state.high_res_masks,
            &state.object_score_logits,
            false,
        )
    }

    pub fn encode_external_memory(
        &self,
        visual: &VisualBackboneOutput,
        high_res_masks: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_points: bool,
    ) -> Result<(Tensor, Tensor)> {
        self.encode_new_memory_from_visual(
            visual,
            high_res_masks,
            object_score_logits,
            is_mask_from_points,
        )
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

fn build_tracker_2d_sine_position_encoding(
    feature: &Tensor,
    num_pos_feats_total: usize,
    normalize: bool,
    scale: f32,
    temperature: f32,
) -> Result<Tensor> {
    let (batch_size, channels, height, width) = feature.dims4()?;
    if channels != num_pos_feats_total {
        candle::bail!(
            "tracker maskmem position encoding expected feature width {num_pos_feats_total}, got {channels}"
        )
    }
    if num_pos_feats_total % 2 != 0 {
        candle::bail!(
            "tracker maskmem position encoding requires even width, got {num_pos_feats_total}"
        )
    }
    let num_pos_feats = num_pos_feats_total / 2;
    let device = feature.device();
    let dtype = feature.dtype();
    let eps = 1e-6f32;
    let mut dim_t = Vec::with_capacity(num_pos_feats);
    for idx in 0..num_pos_feats {
        let exponent = 2.0 * (idx / 2) as f32 / num_pos_feats as f32;
        dim_t.push(temperature.powf(exponent));
    }
    let mut encoding = vec![0f32; num_pos_feats_total * height * width];
    for y in 0..height {
        let mut y_embed = (y + 1) as f32;
        if normalize {
            y_embed = y_embed / (height as f32 + eps) * scale;
        }
        for x in 0..width {
            let mut x_embed = (x + 1) as f32;
            if normalize {
                x_embed = x_embed / (width as f32 + eps) * scale;
            }
            for pair_idx in 0..(num_pos_feats / 2) {
                let even_idx = pair_idx * 2;
                let odd_idx = even_idx + 1;
                let y_even = y_embed / dim_t[even_idx];
                let y_odd = y_embed / dim_t[odd_idx];
                let x_even = x_embed / dim_t[even_idx];
                let x_odd = x_embed / dim_t[odd_idx];
                let spatial_index = y * width + x;
                encoding[even_idx * height * width + spatial_index] = y_even.sin();
                encoding[odd_idx * height * width + spatial_index] = y_odd.cos();
                encoding[(num_pos_feats + even_idx) * height * width + spatial_index] =
                    x_even.sin();
                encoding[(num_pos_feats + odd_idx) * height * width + spatial_index] = x_odd.cos();
            }
        }
    }
    let encoding = Tensor::from_slice(&encoding, (1, num_pos_feats_total, height, width), device)?;
    encoding.repeat((batch_size, 1, 1, 1))?.to_dtype(dtype)
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

    Tensor::from_vec(output, (batch, channels, out_h, out_w), &Device::Cpu)?
        .to_device(input.device())
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
        #[serde(default = "default_recent_occlusion_suppression_threshold")]
        suppress_overlapping_based_on_recent_occlusion_threshold: f32,
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

    fn default_recent_occlusion_suppression_threshold() -> f32 {
        0.7
    }

    #[derive(Debug, Clone, Copy)]
    enum TrackerFixtureBundle {
        Default,
        TemporalDisambiguation,
        LongHistoryStride1,
        LongHistoryObjPtrOverflow,
        LongHistoryStrideGt1,
        LongHistoryKeepFirstCond,
        LongHistoryTemporalDisambiguation,
        LongHistoryTrimMem,
        PointSingleClick,
        PointMultiClick,
        PointAllPoints,
        MaskDirect,
        MemNonOverlap,
        OffloadOutputCpu,
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
                Self::LongHistoryStride1 => {
                    "../candle-examples/examples/sam3/reference_video_long_history_stride1_debug/debug"
                }
                Self::LongHistoryObjPtrOverflow => {
                    "../candle-examples/examples/sam3/reference_video_long_history_obj_ptr_overflow_debug/debug"
                }
                Self::LongHistoryStrideGt1 => {
                    "../candle-examples/examples/sam3/reference_video_long_history_stride_gt1_debug/debug"
                }
                Self::LongHistoryKeepFirstCond => {
                    "../candle-examples/examples/sam3/reference_video_long_history_keep_first_cond_debug/debug"
                }
                Self::LongHistoryTemporalDisambiguation => {
                    "../candle-examples/examples/sam3/reference_video_long_history_temporal_disambiguation_debug/debug"
                }
                Self::LongHistoryTrimMem => {
                    "../candle-examples/examples/sam3/reference_video_long_history_trim_mem_debug/debug"
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
                Self::MemNonOverlap => {
                    "../candle-examples/examples/sam3/reference_video_mem_non_overlap_debug/debug"
                }
                Self::OffloadOutputCpu => {
                    "../candle-examples/examples/sam3/reference_video_offload_output_cpu_debug/debug"
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
        let tensors =
            unsafe { candle::safetensors::MmapedSafetensors::new(&path) }.map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to mmap tracker fixture tensors {}: {err}",
                    path.display()
                ))
            })?;
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
        config.image_size = fixture.image_size;
        config.backbone_stride = fixture.backbone_stride;
        config.num_maskmem = fixture.num_maskmem;
        config.max_cond_frames_in_attn = fixture.max_cond_frames_in_attn;
        config.keep_first_cond_frame = fixture.keep_first_cond_frame;
        config.memory_temporal_stride_for_eval = fixture.memory_temporal_stride_for_eval;
        config.max_obj_ptrs_in_encoder = fixture.max_obj_ptrs_in_encoder;
        config.non_overlap_masks_for_mem_enc = fixture.non_overlap_masks_for_mem_enc;
        config.sigmoid_scale_for_mem_enc = fixture.sigmoid_scale_for_mem_enc;
        config.sigmoid_bias_for_mem_enc = fixture.sigmoid_bias_for_mem_enc;
        config.mf_threshold = fixture.mf_threshold;
        config.multimask_output_in_sam = fixture.multimask_output_in_sam;
        config.multimask_output_for_tracking = fixture.multimask_output_for_tracking;
        config.multimask_min_pt_num = fixture.multimask_min_pt_num;
        config.multimask_max_pt_num = fixture.multimask_max_pt_num;
        config.mask_decoder.dynamic_multimask_via_stability = fixture
            .sam_mask_decoder_extra_args
            .dynamic_multimask_via_stability;
        config.mask_decoder.dynamic_multimask_stability_delta = fixture
            .sam_mask_decoder_extra_args
            .dynamic_multimask_stability_delta;
        config.mask_decoder.dynamic_multimask_stability_thresh = fixture
            .sam_mask_decoder_extra_args
            .dynamic_multimask_stability_thresh;
        config.predictor.with_backbone = false;
        config.predictor.forward_backbone_per_frame_for_eval =
            fixture.forward_backbone_per_frame_for_eval;
        config.predictor.trim_past_non_cond_mem_for_eval = fixture.trim_past_non_cond_mem_for_eval;
        config.predictor.offload_output_to_cpu_for_eval = fixture.offload_output_to_cpu_for_eval;
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
        config
            .predictor
            .refinement_detector_cond_frame_removal_window =
            predictor.refinement_detector_cond_frame_removal_window;
        config.predictor.hotstart_delay = predictor.hotstart_delay;
        config.predictor.hotstart_unmatch_thresh = predictor.hotstart_unmatch_thresh;
        config.predictor.hotstart_dup_thresh = predictor.hotstart_dup_thresh;
        config
            .predictor
            .suppress_overlapping_based_on_recent_occlusion_threshold =
            predictor.suppress_overlapping_based_on_recent_occlusion_threshold;
        config.predictor.masklet_confirmation_enable = predictor.masklet_confirmation_enable;
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
        let (feat0_key, feat1_key, feat2_key, pos0_key, pos1_key, pos2_key) = if forward_stage
            .tensor_keys
            .contains_key("high_res_features.0")
        {
            (
                "high_res_features.0",
                "high_res_features.1",
                "backbone_features",
                None,
                None,
                None,
            )
        } else {
            (
                "forward_image_output.backbone_fpn.0",
                "forward_image_output.backbone_fpn.1",
                "forward_image_output.backbone_fpn.2",
                Some("forward_image_output.vision_pos_enc.0"),
                Some("forward_image_output.vision_pos_enc.1"),
                Some("forward_image_output.vision_pos_enc.2"),
            )
        };
        let high_res_0 =
            load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys[feat0_key].as_str())?;
        let high_res_1 =
            load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys[feat1_key].as_str())?;
        let backbone =
            load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys[feat2_key].as_str())?;
        let pos0 = match pos0_key {
            Some(key) => {
                load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys[key].as_str())?
            }
            None => Tensor::zeros(high_res_0.shape(), high_res_0.dtype(), &candle::Device::Cpu)?,
        };
        let pos1 = match pos1_key {
            Some(key) => {
                load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys[key].as_str())?
            }
            None => Tensor::zeros(high_res_1.shape(), high_res_1.dtype(), &candle::Device::Cpu)?,
        };
        let pos2 = match pos2_key {
            Some(key) => {
                load_tracker_fixture_tensor(bundle, forward_stage.tensor_keys[key].as_str())?
            }
            None => Tensor::zeros(backbone.shape(), backbone.dtype(), &candle::Device::Cpu)?,
        };
        Ok(VisualBackboneOutput {
            backbone_fpn: vec![high_res_0, high_res_1, backbone],
            vision_pos_enc: vec![pos0, pos1, pos2],
            sam2_backbone_fpn: None,
            sam2_pos_enc: None,
        })
    }

    fn assert_tensor_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        atol: f32,
    ) -> Result<()> {
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
            candle::bail!("{label} max abs diff {max_abs_diff:.6} exceeded tolerance {atol:.6}");
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
        let point_coords = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["point_inputs.point_coords"].as_str(),
        )?;
        let point_labels = load_tracker_fixture_tensor(
            bundle,
            forward_stage.tensor_keys["point_inputs.point_labels"].as_str(),
        )?;
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
        let (low_res_multimasks, ious, sam_output_tokens, object_score_logits) =
            model.sam_mask_decoder.forward(
                &image_embeddings,
                &image_pe,
                &sparse_prompt_embeddings,
                &dense_prompt_embeddings,
                stage.metadata["multimask_output"]
                    .as_bool()
                    .unwrap_or(false),
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
        let backbone_features =
            load_tracker_fixture_tensor(bundle, stage.tensor_keys["backbone_features"].as_str())?
                .to_dtype(DType::F32)?;
        let point_prompt = if stage.metadata["has_point_inputs"]
            .as_bool()
            .unwrap_or(false)
        {
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
            Some(
                load_tracker_fixture_tensor(bundle, stage.tensor_keys["mask_inputs"].as_str())?
                    .to_dtype(DType::F32)?,
            )
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
            stage.metadata["multimask_output"]
                .as_bool()
                .unwrap_or(false),
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

    fn maybe_tracker_record<'a>(
        manifest: &'a TrackerInternalManifest,
        frame_idx: usize,
        stage: &str,
    ) -> Option<&'a TrackerInternalRecord> {
        manifest
            .records
            .iter()
            .find(|record| record.frame_idx == frame_idx && record.stage == stage)
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

    fn metadata_usize_vec(metadata: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
        metadata
            .get(key)
            .and_then(|value| value.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "tracker fixture metadata missing usize vec `{key}`"
                ))
            })?
            .iter()
            .map(|value| {
                value.as_u64().map(|value| value as usize).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "tracker fixture metadata `{key}` contained non-usize value {value}"
                    ))
                })
            })
            .collect()
    }

    fn metadata_i64_vec(metadata: &serde_json::Value, key: &str) -> Result<Vec<i64>> {
        metadata
            .get(key)
            .and_then(|value| value.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!("tracker fixture metadata missing i64 vec `{key}`"))
            })?
            .iter()
            .map(|value| {
                value.as_i64().ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "tracker fixture metadata `{key}` contained non-i64 value {value}"
                    ))
                })
            })
            .collect()
    }

    fn load_track_step_history_state(
        bundle: TrackerFixtureBundle,
        manifest: &TrackerInternalManifest,
        frame_idx: usize,
        dtype: DType,
    ) -> Result<Option<TrackerFrameState>> {
        let Some(track_step) = maybe_tracker_record(manifest, frame_idx, "track_step") else {
            return Ok(None);
        };
        let device = &candle::Device::Cpu;
        let is_cond_frame = track_step.metadata["is_init_cond_frame"]
            .as_bool()
            .unwrap_or(false);
        let low_res_masks = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["track_step_output.pred_masks"].as_str(),
        )?
        .to_dtype(dtype)?;
        let high_res_masks = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["track_step_output.pred_masks_high_res"].as_str(),
        )?
        .to_dtype(dtype)?;
        let obj_ptr = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["track_step_output.obj_ptr"].as_str(),
        )?
        .to_dtype(dtype)?;
        let object_score_logits = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["track_step_output.object_score_logits"].as_str(),
        )?
        .to_dtype(dtype)?;
        let iou_scores = match track_step.tensor_keys.get("track_step_output.iou_score") {
            Some(key) => load_tracker_fixture_tensor(bundle, key.as_str())?.to_dtype(dtype)?,
            None => Tensor::zeros((low_res_masks.dim(0)?, 1), dtype, device)?,
        };
        let maskmem_features = track_step
            .tensor_keys
            .get("track_step_output.maskmem_features")
            .map(|key| load_tracker_fixture_tensor(bundle, key.as_str()))
            .transpose()?;
        let maskmem_features = maskmem_features
            .map(|tensor| tensor.to_dtype(dtype))
            .transpose()?;
        let maskmem_pos_enc = track_step
            .tensor_keys
            .get("track_step_output.maskmem_pos_enc.0")
            .map(|key| load_tracker_fixture_tensor(bundle, key.as_str()))
            .transpose()?;
        let maskmem_pos_enc = maskmem_pos_enc
            .map(|tensor| tensor.to_dtype(dtype))
            .transpose()?;
        if maskmem_features.is_some() || !is_cond_frame {
            return Ok(Some(TrackerFrameState {
                low_res_masks,
                high_res_masks,
                iou_scores,
                obj_ptr,
                object_score_logits,
                maskmem_features,
                maskmem_pos_enc,
                is_cond_frame,
            }));
        }

        let Some(preflight) =
            maybe_tracker_record(manifest, frame_idx, "propagate_in_video_preflight")
        else {
            return Ok(Some(TrackerFrameState {
                low_res_masks,
                high_res_masks,
                iou_scores,
                obj_ptr,
                object_score_logits,
                maskmem_features: None,
                maskmem_pos_enc: None,
                is_cond_frame,
            }));
        };
        let features_key =
            format!("preflight_output.cond_frame_outputs.{frame_idx}.maskmem_features");
        let pos_key = format!("preflight_output.cond_frame_outputs.{frame_idx}.maskmem_pos_enc.0");
        let maskmem_features = preflight
            .tensor_keys
            .get(&features_key)
            .map(|key| load_tracker_fixture_tensor(bundle, key.as_str()))
            .transpose()?;
        let maskmem_features = maskmem_features
            .map(|tensor| tensor.to_dtype(dtype))
            .transpose()?;
        let maskmem_pos_enc = preflight
            .tensor_keys
            .get(&pos_key)
            .map(|key| load_tracker_fixture_tensor(bundle, key.as_str()))
            .transpose()?;
        let maskmem_pos_enc = maskmem_pos_enc
            .map(|tensor| tensor.to_dtype(dtype))
            .transpose()?;
        Ok(Some(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores,
            obj_ptr,
            object_score_logits,
            maskmem_features,
            maskmem_pos_enc,
            is_cond_frame,
        }))
    }

    fn load_prepare_selected_conditioning_state(
        bundle: TrackerFixtureBundle,
        prepare: &TrackerInternalRecord,
        frame_idx: usize,
        image_size: usize,
        dtype: DType,
    ) -> Result<Option<TrackerFrameState>> {
        let pred_key = format!("selected_conditioning_frames.{frame_idx}.pred_masks");
        let Some(pred_key_ref) = prepare.tensor_keys.get(&pred_key) else {
            return Ok(None);
        };
        let device = &candle::Device::Cpu;
        let low_res_masks =
            load_tracker_fixture_tensor(bundle, pred_key_ref.as_str())?.to_dtype(dtype)?;
        let iou_key = format!("selected_conditioning_frames.{frame_idx}.iou_score");
        let iou_scores = match prepare.tensor_keys.get(&iou_key) {
            Some(key) => load_tracker_fixture_tensor(bundle, key.as_str())?.to_dtype(dtype)?,
            None => Tensor::zeros((low_res_masks.dim(0)?, 1), dtype, device)?,
        };
        let obj_ptr = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_conditioning_frames.{frame_idx}.obj_ptr").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let object_score_logits = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_conditioning_frames.{frame_idx}.object_score_logits").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let maskmem_features = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_conditioning_frames.{frame_idx}.maskmem_features").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let maskmem_pos_enc = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_conditioning_frames.{frame_idx}.maskmem_pos_enc.0").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let high_res_masks = Tensor::zeros(
            (low_res_masks.dim(0)?, 1, image_size, image_size),
            dtype,
            device,
        )?;
        Ok(Some(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores,
            obj_ptr,
            object_score_logits,
            maskmem_features: Some(maskmem_features),
            maskmem_pos_enc: Some(maskmem_pos_enc),
            is_cond_frame: true,
        }))
    }

    fn load_prepare_selected_memory_state(
        bundle: TrackerFixtureBundle,
        prepare: &TrackerInternalRecord,
        frame_idx: usize,
        image_size: usize,
        dtype: DType,
    ) -> Result<Option<TrackerFrameState>> {
        let pred_key = format!("selected_memory_frames.{frame_idx}.pred_masks");
        let Some(pred_key_ref) = prepare.tensor_keys.get(&pred_key) else {
            return Ok(None);
        };
        let device = &candle::Device::Cpu;
        let low_res_masks =
            load_tracker_fixture_tensor(bundle, pred_key_ref.as_str())?.to_dtype(dtype)?;
        let iou_key = format!("selected_memory_frames.{frame_idx}.iou_score");
        let iou_scores = match prepare.tensor_keys.get(&iou_key) {
            Some(key) => load_tracker_fixture_tensor(bundle, key.as_str())?.to_dtype(dtype)?,
            None => Tensor::zeros((low_res_masks.dim(0)?, 1), dtype, device)?,
        };
        let obj_ptr = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys[format!("selected_memory_frames.{frame_idx}.obj_ptr").as_str()]
                .as_str(),
        )?
        .to_dtype(dtype)?;
        let object_score_logits = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_memory_frames.{frame_idx}.object_score_logits").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let maskmem_features = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_memory_frames.{frame_idx}.maskmem_features").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let maskmem_pos_enc = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys
                [format!("selected_memory_frames.{frame_idx}.maskmem_pos_enc.0").as_str()]
            .as_str(),
        )?
        .to_dtype(dtype)?;
        let high_res_masks = Tensor::zeros(
            (low_res_masks.dim(0)?, 1, image_size, image_size),
            dtype,
            device,
        )?;
        Ok(Some(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores,
            obj_ptr,
            object_score_logits,
            maskmem_features: Some(maskmem_features),
            maskmem_pos_enc: Some(maskmem_pos_enc),
            is_cond_frame: false,
        }))
    }

    fn build_history_for_prepare_frame(
        bundle: TrackerFixtureBundle,
        manifest: &TrackerInternalManifest,
        prepare: &TrackerInternalRecord,
        target_frame_idx: usize,
        image_size: usize,
        dtype: DType,
    ) -> Result<BTreeMap<usize, TrackerFrameState>> {
        let selected_cond =
            metadata_usize_vec(&prepare.metadata, "selected_conditioning_frame_indices")?;
        let unselected_cond = prepare
            .metadata
            .get("unselected_conditioning_frame_indices")
            .and_then(|value| value.as_array())
            .map(|_| metadata_usize_vec(&prepare.metadata, "unselected_conditioning_frame_indices"))
            .transpose()?
            .unwrap_or_default();
        let selected_memory =
            metadata_usize_vec(&prepare.metadata, "selected_memory_frame_indices")?;
        let selected_object_pointer_frames =
            metadata_usize_vec(&prepare.metadata, "selected_object_pointer_frame_indices")?;
        let mut history = BTreeMap::new();

        for frame_idx in 0..target_frame_idx {
            if let Some(state) = load_track_step_history_state(bundle, manifest, frame_idx, dtype)?
            {
                history.insert(frame_idx, state);
            }
        }

        for frame_idx in selected_cond.iter().copied() {
            if let Some(state) = load_prepare_selected_conditioning_state(
                bundle, prepare, frame_idx, image_size, dtype,
            )? {
                history.insert(frame_idx, state);
            }
        }

        for frame_idx in unselected_cond.iter().copied() {
            if history.contains_key(&frame_idx) {
                continue;
            }
            if let Some(state) = load_track_step_history_state(bundle, manifest, frame_idx, dtype)?
            {
                history.insert(frame_idx, state);
            }
        }

        for frame_idx in selected_memory.iter().copied() {
            if history.contains_key(&frame_idx) {
                continue;
            }
            if let Some(state) =
                load_prepare_selected_memory_state(bundle, prepare, frame_idx, image_size, dtype)?
            {
                history.insert(frame_idx, state);
            }
        }

        for frame_idx in selected_object_pointer_frames.iter().copied() {
            if history.contains_key(&frame_idx) {
                continue;
            }
            let key = format!("selected_object_pointer_frames.{frame_idx}.obj_ptr");
            let Some(obj_ptr_key) = prepare.tensor_keys.get(&key) else {
                continue;
            };
            let device = &candle::Device::Cpu;
            let obj_ptr =
                load_tracker_fixture_tensor(bundle, obj_ptr_key.as_str())?.to_dtype(dtype)?;
            let low_res_masks = Tensor::zeros((obj_ptr.dim(0)?, 1, 1, 1), dtype, device)?;
            let high_res_masks =
                Tensor::zeros((obj_ptr.dim(0)?, 1, image_size, image_size), dtype, device)?;
            let iou_scores = Tensor::zeros((obj_ptr.dim(0)?, 1), dtype, device)?;
            let object_score_logits = Tensor::zeros((obj_ptr.dim(0)?, 1), dtype, device)?;
            history.insert(
                frame_idx,
                TrackerFrameState {
                    low_res_masks,
                    high_res_masks,
                    iou_scores,
                    obj_ptr,
                    object_score_logits,
                    maskmem_features: None,
                    maskmem_pos_enc: None,
                    is_cond_frame: false,
                },
            );
        }

        Ok(history)
    }

    fn assert_prepare_memory_conditioned_features_fixture_matches(
        bundle: TrackerFixtureBundle,
        frame_idx: usize,
        pix_feat_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let track_step = tracker_record(&manifest, frame_idx, "track_step")?;
        let prepare = tracker_record(&manifest, frame_idx, "prepare_memory_conditioned_features")?;
        let compute_dtype = model.no_obj_ptr.dtype();
        let current_vision_feats = vec![load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_feats"].as_str(),
        )?
        .to_dtype(compute_dtype)?];
        let current_vision_pos_embeds = vec![load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_pos_embeds"].as_str(),
        )?
        .to_dtype(compute_dtype)?];
        let history = build_history_for_prepare_frame(
            bundle,
            &manifest,
            prepare,
            frame_idx,
            model.config.image_size,
            compute_dtype,
        )?;
        let actual = model.prepare_memory_conditioned_features(
            frame_idx,
            false,
            current_vision_feats.as_slice(),
            current_vision_pos_embeds.as_slice(),
            &[(
                model.config.image_embedding_size(),
                model.config.image_embedding_size(),
            )],
            &history,
            30,
            false,
            true,
        )?;
        let expected_cond =
            metadata_usize_vec(&prepare.metadata, "selected_conditioning_frame_indices")?;
        let expected_mem = metadata_usize_vec(&prepare.metadata, "selected_memory_frame_indices")?;
        let expected_ptr =
            metadata_usize_vec(&prepare.metadata, "selected_object_pointer_frame_indices")?;
        assert_eq!(actual.selected_conditioning_frame_indices, expected_cond);
        assert_eq!(actual.selected_memory_frame_indices, expected_mem);
        assert_eq!(actual.selected_object_pointer_frame_indices, expected_ptr);

        let expected_pix =
            load_tracker_fixture_tensor(bundle, prepare.tensor_keys["pix_feat_with_mem"].as_str())?;
        assert_tensor_close(
            "prepare_memory_conditioned_features.pix_feat_with_mem",
            &actual.pix_feat_with_mem,
            &expected_pix,
            pix_feat_atol,
        )?;

        let offsets = metadata_i64_vec(
            &prepare.metadata,
            "selected_object_pointer_temporal_offsets",
        )?;
        let max_abs_pos = prepare.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "prepare_memory_conditioned_features missing max_obj_ptrs_in_encoder".into(),
                )
            })? as usize;
        let expected_pos = load_tracker_fixture_tensor(
            bundle,
            prepare.tensor_keys["object_pointer_temporal_pos_enc"].as_str(),
        )?;
        let actual_pos = model.get_tpos_enc(
            offsets.as_slice(),
            &candle::Device::Cpu,
            Some(max_abs_pos),
            false,
        )?;
        assert_tensor_close(
            "prepare_memory_conditioned_features.object_pointer_temporal_pos_enc",
            &actual_pos,
            &expected_pos,
            2e-2,
        )?;
        Ok(())
    }

    fn assert_memory_conditioning_prompt_fixture_matches(
        bundle: TrackerFixtureBundle,
        frame_idx: usize,
        prompt_atol: f32,
        prompt_pos_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let prepare = tracker_record(&manifest, frame_idx, "prepare_memory_conditioned_features")?;
        let encoder = tracker_record(&manifest, frame_idx, "memory_transformer_encoder")?;
        let history = build_history_for_prepare_frame(
            bundle,
            &manifest,
            prepare,
            frame_idx,
            model.config.image_size,
            model.no_obj_ptr.dtype(),
        )?;
        let cond_frame_outputs = history
            .iter()
            .filter_map(|(frame, state)| state.is_cond_frame.then_some((*frame, state)))
            .collect::<BTreeMap<_, _>>();
        let prepared = model.build_memory_conditioning_prompt(
            frame_idx,
            &history,
            30,
            false,
            &cond_frame_outputs,
        )?;
        let expected_prompt =
            load_tracker_fixture_tensor(bundle, encoder.tensor_keys["prompt"].as_str())?;
        let expected_prompt_pos =
            load_tracker_fixture_tensor(bundle, encoder.tensor_keys["prompt_pos"].as_str())?;
        let actual_prompt = prepared
            .prompt
            .ok_or_else(|| candle::Error::Msg("expected prompt tensor for fixture".into()))?;
        let actual_prompt_pos = prepared
            .prompt_pos
            .ok_or_else(|| candle::Error::Msg("expected prompt_pos tensor for fixture".into()))?;
        assert_eq!(
            prepared.num_obj_ptr_tokens,
            encoder.metadata["num_obj_ptr_tokens"].as_u64().unwrap_or(0) as usize
        );
        assert_tensor_close(
            "memory_conditioning_prompt.prompt",
            &actual_prompt,
            &expected_prompt,
            prompt_atol,
        )?;
        assert_tensor_close(
            "memory_conditioning_prompt.prompt_pos",
            &actual_prompt_pos,
            &expected_prompt_pos,
            prompt_pos_atol,
        )?;
        Ok(())
    }

    fn assert_memory_transformer_encoder_fixture_matches(
        bundle: TrackerFixtureBundle,
        frame_idx: usize,
        memory_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let track_step = tracker_record(&manifest, frame_idx, "track_step")?;
        let encoder = tracker_record(&manifest, frame_idx, "memory_transformer_encoder")?;
        let src = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_feats"].as_str(),
        )?
        .to_dtype(model.no_obj_ptr.dtype())?
        .transpose(0, 1)?;
        let src_pos = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_pos_embeds"].as_str(),
        )?
        .to_dtype(model.no_obj_ptr.dtype())?
        .transpose(0, 1)?;
        let prompt = load_tracker_fixture_tensor(bundle, encoder.tensor_keys["prompt"].as_str())?
            .to_dtype(model.no_obj_ptr.dtype())?
            .transpose(0, 1)?;
        let prompt_pos =
            load_tracker_fixture_tensor(bundle, encoder.tensor_keys["prompt_pos"].as_str())?
                .to_dtype(model.no_obj_ptr.dtype())?
                .transpose(0, 1)?;
        let expected_memory = load_tracker_fixture_tensor(
            bundle,
            encoder.tensor_keys["memory_transformer_encoder_output.memory"].as_str(),
        )?;
        let actual = model.memory_transformer.forward(
            &src,
            &prompt,
            Some(&src_pos),
            Some(&prompt_pos),
            encoder.metadata["num_obj_ptr_tokens"].as_u64().unwrap_or(0) as usize,
        )?;
        let actual = actual.transpose(0, 1)?;
        assert_tensor_close(
            "memory_transformer_encoder.memory",
            &actual,
            &expected_memory,
            memory_atol,
        )?;
        Ok(())
    }

    fn assert_memory_transformer_encoder_from_reconstructed_prompt_fixture_matches(
        bundle: TrackerFixtureBundle,
        frame_idx: usize,
        memory_atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let track_step = tracker_record(&manifest, frame_idx, "track_step")?;
        let prepare = tracker_record(&manifest, frame_idx, "prepare_memory_conditioned_features")?;
        let encoder = tracker_record(&manifest, frame_idx, "memory_transformer_encoder")?;
        let history = build_history_for_prepare_frame(
            bundle,
            &manifest,
            prepare,
            frame_idx,
            model.config.image_size,
            model.no_obj_ptr.dtype(),
        )?;
        let cond_frame_outputs = history
            .iter()
            .filter_map(|(frame, state)| state.is_cond_frame.then_some((*frame, state)))
            .collect::<BTreeMap<_, _>>();
        let prepared = model.build_memory_conditioning_prompt(
            frame_idx,
            &history,
            30,
            false,
            &cond_frame_outputs,
        )?;
        let src = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_feats"].as_str(),
        )?
        .to_dtype(model.no_obj_ptr.dtype())?
        .transpose(0, 1)?;
        let src_pos = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_pos_embeds"].as_str(),
        )?
        .to_dtype(model.no_obj_ptr.dtype())?
        .transpose(0, 1)?;
        let prompt = prepared
            .prompt
            .ok_or_else(|| candle::Error::Msg("expected reconstructed prompt".into()))?
            .to_dtype(model.no_obj_ptr.dtype())?
            .transpose(0, 1)?;
        let prompt_pos = prepared
            .prompt_pos
            .ok_or_else(|| candle::Error::Msg("expected reconstructed prompt_pos".into()))?
            .to_dtype(model.no_obj_ptr.dtype())?
            .transpose(0, 1)?;
        let expected_memory = load_tracker_fixture_tensor(
            bundle,
            encoder.tensor_keys["memory_transformer_encoder_output.memory"].as_str(),
        )?;
        let actual = model.memory_transformer.forward(
            &src,
            &prompt,
            Some(&src_pos),
            Some(&prompt_pos),
            prepared.num_obj_ptr_tokens,
        )?;
        let actual = actual.transpose(0, 1)?;
        assert_tensor_close(
            "memory_transformer_encoder.reconstructed_prompt.memory",
            &actual,
            &expected_memory,
            memory_atol,
        )?;
        Ok(())
    }

    fn tracker_record_with_bool_metadata<'a>(
        manifest: &'a TrackerInternalManifest,
        frame_idx: usize,
        stage: &str,
        key: &str,
        expected: bool,
    ) -> Result<&'a TrackerInternalRecord> {
        manifest
            .records
            .iter()
            .find(|record| {
                record.frame_idx == frame_idx
                    && record.stage == stage
                    && record.metadata.get(key).and_then(|value| value.as_bool()) == Some(expected)
            })
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "missing tracker record stage={stage} frame={frame_idx} with {key}={expected}"
                ))
            })
    }

    fn build_top_level_visual_from_track_step(
        bundle: TrackerFixtureBundle,
        manifest: &TrackerInternalManifest,
        frame_idx: usize,
        image_embedding_size: usize,
    ) -> Result<VisualBackboneOutput> {
        let track_step = tracker_record(manifest, frame_idx, "track_step")?;
        let current_vision_feats = load_tracker_fixture_tensor(
            bundle,
            track_step.tensor_keys["current_vision_feats"].as_str(),
        )?;
        let (hw, batch_size, channels) = current_vision_feats.dims3()?;
        if hw != image_embedding_size * image_embedding_size {
            candle::bail!(
                "track_step current_vision_feats length {hw} does not match expected top-level area {}",
                image_embedding_size * image_embedding_size
            );
        }
        let backbone = current_vision_feats
            .transpose(0, 1)?
            .transpose(1, 2)?
            .reshape((
                batch_size,
                channels,
                image_embedding_size,
                image_embedding_size,
            ))?;
        let pos = Tensor::zeros(backbone.shape(), backbone.dtype(), &candle::Device::Cpu)?;
        Ok(VisualBackboneOutput {
            backbone_fpn: vec![backbone],
            vision_pos_enc: vec![pos],
            sam2_backbone_fpn: None,
            sam2_pos_enc: None,
        })
    }

    fn assert_encode_external_memory_fixture_matches(
        bundle: TrackerFixtureBundle,
        frame_idx: usize,
        is_mask_from_pts: bool,
        atol: f32,
    ) -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(bundle)? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(bundle)?;
        let run_memory_encoder = tracker_record_with_bool_metadata(
            &manifest,
            frame_idx,
            "run_memory_encoder",
            "is_mask_from_pts",
            is_mask_from_pts,
        )?;
        let encode_new_memory = tracker_record_with_bool_metadata(
            &manifest,
            frame_idx,
            "encode_new_memory",
            "is_mask_from_pts",
            is_mask_from_pts,
        )?;
        let visual = build_top_level_visual_from_track_step(
            bundle,
            &manifest,
            frame_idx,
            model.config.image_embedding_size(),
        )?;
        let high_res_masks = load_tracker_fixture_tensor(
            bundle,
            run_memory_encoder.tensor_keys["high_res_masks"].as_str(),
        )?;
        let object_score_logits = load_tracker_fixture_tensor(
            bundle,
            run_memory_encoder.tensor_keys["object_score_logits"].as_str(),
        )?;
        let (actual_features, actual_pos_enc) = model.encode_external_memory(
            &visual,
            &high_res_masks,
            &object_score_logits,
            is_mask_from_pts,
        )?;
        let expected_features = load_tracker_fixture_tensor(
            bundle,
            encode_new_memory.tensor_keys["maskmem_features"].as_str(),
        )?;
        let expected_pos_enc = load_tracker_fixture_tensor(
            bundle,
            encode_new_memory.tensor_keys["maskmem_pos_enc.0"].as_str(),
        )?;
        let bf16_backend_limited =
            expected_features.dtype() == DType::BF16 && !actual_features.device().supports_bf16();
        if bf16_backend_limited {
            if actual_features.shape() != expected_features.shape() {
                candle::bail!(
                    "encode_external_memory.maskmem_features shape mismatch under BF16 backend limitation: actual {:?}, expected {:?}",
                    actual_features.shape().dims(),
                    expected_features.shape().dims()
                );
            }
            if actual_pos_enc.shape() != expected_pos_enc.shape() {
                candle::bail!(
                    "encode_external_memory.maskmem_pos_enc shape mismatch under BF16 backend limitation: actual {:?}, expected {:?}",
                    actual_pos_enc.shape().dims(),
                    expected_pos_enc.shape().dims()
                );
            }
            return Ok(());
        }
        assert_tensor_close(
            "encode_external_memory.maskmem_pos_enc",
            &actual_pos_enc,
            &expected_pos_enc,
            atol,
        )?;
        assert_tensor_close(
            "encode_external_memory.maskmem_features",
            &actual_features,
            &expected_features,
            atol,
        )?;
        Ok(())
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

    #[test]
    fn fixture_backed_tracker_config_matches_mem_non_overlap_runtime_upstream_bundle() -> Result<()>
    {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MemNonOverlap)?;
        let config = tracker_runtime_config_from_fixture_manifest(&manifest);
        assert!(config.non_overlap_masks_for_mem_enc);
        assert!(!config.predictor.trim_past_non_cond_mem_for_eval);
        assert!(!config.predictor.offload_output_to_cpu_for_eval);
        Ok(())
    }

    #[test]
    fn fixture_backed_tracker_config_matches_long_history_trim_mem_runtime_upstream_bundle(
    ) -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryTrimMem)?;
        let config = tracker_runtime_config_from_fixture_manifest(&manifest);
        assert!(!config.non_overlap_masks_for_mem_enc);
        assert!(config.predictor.trim_past_non_cond_mem_for_eval);
        assert!(!config.predictor.offload_output_to_cpu_for_eval);
        Ok(())
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
        assert_eq!(
            track_stage.metadata["has_mask_inputs"].as_bool(),
            Some(true)
        );
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
        let actual = model.get_tpos_enc(
            offsets.as_slice(),
            &candle::Device::Cpu,
            Some(max_abs_pos),
            false,
        )?;
        assert_tensor_close("get_tpos_enc", &actual, &expected, 1e-2)?;
        Ok(())
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_stride1_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_stride_gt1_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryStrideGt1,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_obj_ptr_overflow_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryObjPtrOverflow,
            29,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_keep_first_cond_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryKeepFirstCond,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_temporal_disambiguation_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryTemporalDisambiguation,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_get_tpos_enc_matches_stride1_long_history_fixture_values() -> Result<()> {
        let Some(model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::LongHistoryStride1)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryStride1)?;
        let stage = tracker_record(&manifest, 28, "prepare_memory_conditioned_features")?;
        let offsets =
            metadata_i64_vec(&stage.metadata, "selected_object_pointer_temporal_offsets")?;
        let max_abs_pos = stage.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg("stride1 fixture missing max_obj_ptrs_in_encoder".into())
            })? as usize;
        let expected = load_tracker_fixture_tensor(
            TrackerFixtureBundle::LongHistoryStride1,
            stage.tensor_keys["object_pointer_temporal_pos_enc"].as_str(),
        )?;
        let actual = model.get_tpos_enc(
            offsets.as_slice(),
            &candle::Device::Cpu,
            Some(max_abs_pos),
            false,
        )?;
        assert_tensor_close("get_tpos_enc stride1", &actual, &expected, 2e-2)?;
        Ok(())
    }

    #[test]
    fn tracker_memory_conditioning_prompt_matches_stride1_long_history_fixture_values() -> Result<()>
    {
        assert_memory_conditioning_prompt_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1e-4,
            2e-2,
        )
    }

    #[test]
    fn tracker_memory_transformer_encoder_matches_stride1_long_history_fixture_values() -> Result<()>
    {
        assert_memory_transformer_encoder_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1e-1,
        )
    }

    #[test]
    fn tracker_memory_transformer_encoder_from_reconstructed_prompt_matches_stride1_long_history_fixture_values(
    ) -> Result<()> {
        assert_memory_transformer_encoder_from_reconstructed_prompt_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1e-1,
        )
    }

    #[test]
    fn tracker_object_pointer_selection_overflows_and_truncates_at_encoder_cap() -> Result<()> {
        let manifest =
            load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryObjPtrOverflow)?;
        let stage = tracker_record(&manifest, 29, "prepare_memory_conditioned_features")?;
        let max_obj_ptrs = stage.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "object-pointer overflow fixture missing max_obj_ptrs_in_encoder".into(),
                )
            })? as usize;
        let frame_indices =
            metadata_usize_vec(&stage.metadata, "selected_object_pointer_frame_indices")?;
        let is_cond = stage.metadata["selected_object_pointer_is_conditioning"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "object-pointer overflow fixture missing selected_object_pointer_is_conditioning"
                        .into(),
                )
            })?
            .iter()
            .map(|value| value.as_bool().unwrap_or(false))
            .collect::<Vec<_>>();
        if frame_indices.len() != is_cond.len() {
            candle::bail!(
                "object-pointer overflow fixture has mismatched frame/is_cond lengths: {} vs {}",
                frame_indices.len(),
                is_cond.len()
            );
        }
        let non_cond_frames = frame_indices
            .iter()
            .zip(is_cond.iter())
            .filter_map(|(frame_idx, is_cond)| (!*is_cond).then_some(*frame_idx))
            .collect::<Vec<_>>();
        assert!(
            frame_indices.len() > max_obj_ptrs,
            "overflow fixture should exceed cap: selected {} <= cap {}",
            frame_indices.len(),
            max_obj_ptrs
        );
        assert_eq!(
            non_cond_frames.len(),
            max_obj_ptrs.saturating_sub(1),
            "overflow fixture should retain exactly cap-1 non-conditioning object pointers"
        );
        assert!(
            !non_cond_frames.contains(&13),
            "oldest non-conditioning object pointer frame should be truncated once cap is exceeded"
        );
        for expected in 14..=28 {
            assert!(
                non_cond_frames.contains(&expected),
                "expected non-conditioning object pointer frame {expected} to be retained"
            );
        }
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
        )?
        else {
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
    fn tracker_encode_external_memory_matches_default_bundle_frame0_binary_fixture_values(
    ) -> Result<()> {
        assert_encode_external_memory_fixture_matches(TrackerFixtureBundle::Default, 0, true, 1e-4)
    }

    #[test]
    fn tracker_encode_external_memory_matches_default_bundle_frame1_sigmoid_fixture_values(
    ) -> Result<()> {
        assert_encode_external_memory_fixture_matches(TrackerFixtureBundle::Default, 1, false, 1e-4)
    }

    #[test]
    fn tracker_mem_non_overlap_bundle_records_multi_object_preflight_state() -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MemNonOverlap)?;
        let preflight = tracker_record(&manifest, 0, "propagate_in_video_preflight")?;
        assert_eq!(
            preflight.metadata["tracking_has_started_before"],
            serde_json::json!(false)
        );
        assert_eq!(
            preflight.metadata["tracking_has_started_after"],
            serde_json::json!(true)
        );
        assert_eq!(
            preflight.metadata["after_output_frame_keys"]["cond_frame_outputs"],
            serde_json::json!([0])
        );
        assert_eq!(
            preflight.metadata["after_per_obj_output_frame_keys"]["0"]["cond_frame_outputs"],
            serde_json::json!([0])
        );
        assert_eq!(
            preflight.metadata["after_per_obj_output_frame_keys"]["1"]["cond_frame_outputs"],
            serde_json::json!([0])
        );
        let run_memory_encoder = tracker_record(&manifest, 0, "run_memory_encoder")?;
        assert_eq!(
            run_memory_encoder.metadata["batch_size"],
            serde_json::json!(2)
        );
        Ok(())
    }

    #[test]
    fn tracker_long_history_trim_bundle_records_preflight_state_updates() -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryTrimMem)?;
        let preflight = tracker_record(&manifest, 0, "propagate_in_video_preflight")?;
        assert_eq!(
            preflight.metadata["tracking_has_started_before"],
            serde_json::json!(false)
        );
        assert_eq!(
            preflight.metadata["tracking_has_started_after"],
            serde_json::json!(true)
        );
        assert_eq!(
            preflight.metadata["first_ann_frame_idx"],
            serde_json::json!(0)
        );
        assert_eq!(
            preflight.metadata["after_output_frame_keys"]["cond_frame_outputs"],
            serde_json::json!([0, 5, 10, 15, 20])
        );
        assert_eq!(
            preflight.metadata["consolidated_cond_frame_indices"],
            serde_json::json!([0, 5, 10, 15, 20])
        );
        Ok(())
    }

    #[test]
    fn tracker_non_overlap_constraints_keep_only_highest_scoring_object_per_pixel() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::from_sam3_config(&tiny_config()),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let pred_masks = Tensor::from_vec(
            vec![5.0f32, -2.0, 1.0, 3.0, 4.0, 2.0, 7.0, 1.0],
            (2, 1, 2, 2),
            &device,
        )?;
        let actual = model.apply_non_overlapping_constraints(&pred_masks)?;
        let expected = Tensor::from_vec(
            vec![5.0f32, -10.0, -10.0, 3.0, -10.0, 2.0, 7.0, -10.0],
            (2, 1, 2, 2),
            &device,
        )?;
        assert_tensor_close("non_overlap_masks_for_mem_enc", &actual, &expected, 1e-5)?;
        Ok(())
    }

    #[test]
    fn tracker_offload_state_for_eval_moves_mask_memory_to_cpu_bfloat16() -> Result<()> {
        let device = candle::Device::Cpu;
        let mut config = Sam3TrackerConfig::from_sam3_config(&tiny_config());
        config.predictor.offload_output_to_cpu_for_eval = true;
        let model = Sam3TrackerModel::new(&config, VarBuilder::zeros(DType::F32, &device))?;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 56, 56), DType::F32, &device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
            obj_ptr: Tensor::zeros((1, 32), DType::F32, &device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
            maskmem_features: Some(Tensor::zeros((1, 64, 4, 4), DType::F32, &device)?),
            maskmem_pos_enc: Some(Tensor::zeros((1, 64, 4, 4), DType::F32, &device)?),
            is_cond_frame: true,
        };
        let offloaded = model.maybe_offload_state_for_eval(state)?;
        assert_eq!(
            offloaded.maskmem_features.as_ref().unwrap().dtype(),
            DType::BF16
        );
        assert!(matches!(
            offloaded.maskmem_features.as_ref().unwrap().device(),
            &candle::Device::Cpu
        ));
        assert!(matches!(
            offloaded.maskmem_pos_enc.as_ref().unwrap().device(),
            &candle::Device::Cpu
        ));
        Ok(())
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_skips_trimmed_non_cond_memory() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::build_tracker(false),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let compute_dtype = model.no_obj_ptr.dtype();
        let feat_sizes = vec![(
            model.config().image_embedding_size(),
            model.config().image_embedding_size(),
        )];
        let current_vision_feats = vec![Tensor::zeros(
            (
                model.config().image_embedding_size() * model.config().image_embedding_size(),
                1,
                model.config().hidden_dim,
            ),
            compute_dtype,
            &device,
        )?];
        let current_vision_pos_embeds = vec![Tensor::zeros(
            (
                model.config().image_embedding_size() * model.config().image_embedding_size(),
                1,
                model.config().hidden_dim,
            ),
            compute_dtype,
            &device,
        )?];
        let mut history = BTreeMap::new();
        let mut cond_state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 1, 1), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 1, 1), DType::F32, &device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
            obj_ptr: Tensor::zeros((1, model.config().hidden_dim), DType::F32, &device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: true,
        };
        cond_state.maskmem_features = Some(Tensor::zeros(
            (
                1,
                model.config().memory_dim,
                model.config().image_embedding_size(),
                model.config().image_embedding_size(),
            ),
            DType::F32,
            &device,
        )?);
        cond_state.maskmem_pos_enc = Some(Tensor::zeros(
            (
                1,
                model.config().memory_dim,
                model.config().image_embedding_size(),
                model.config().image_embedding_size(),
            ),
            DType::F32,
            &device,
        )?);
        cond_state.is_cond_frame = true;
        let mut trimmed_non_cond = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 1, 1), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 1, 1), DType::F32, &device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
            obj_ptr: Tensor::zeros((1, model.config().hidden_dim), DType::F32, &device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: false,
        };
        trimmed_non_cond.is_cond_frame = false;
        history.insert(0, cond_state);
        history.insert(1, trimmed_non_cond);
        let prepared = model.prepare_memory_conditioned_features(
            2,
            false,
            current_vision_feats.as_slice(),
            current_vision_pos_embeds.as_slice(),
            feat_sizes.as_slice(),
            &history,
            3,
            false,
            true,
        )?;
        assert_eq!(prepared.selected_conditioning_frame_indices, vec![0]);
        assert!(prepared.selected_memory_frame_indices.is_empty());
        assert_eq!(prepared.selected_object_pointer_frame_indices, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn tracker_history_builder_recovers_selected_long_history_frames() -> Result<()> {
        let Some(model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::LongHistoryStride1)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryStride1)?;
        let prepare = tracker_record(&manifest, 28, "prepare_memory_conditioned_features")?;
        let history = build_history_for_prepare_frame(
            TrackerFixtureBundle::LongHistoryStride1,
            &manifest,
            prepare,
            28,
            model.config.image_size,
            model.no_obj_ptr.dtype(),
        )?;
        for frame_idx in
            metadata_usize_vec(&prepare.metadata, "selected_conditioning_frame_indices")?
        {
            assert!(history.contains_key(&frame_idx));
            assert!(history.get(&frame_idx).unwrap().is_cond_frame);
        }
        for frame_idx in metadata_usize_vec(&prepare.metadata, "selected_memory_frame_indices")? {
            assert!(history.contains_key(&frame_idx));
            assert!(!history.get(&frame_idx).unwrap().is_cond_frame);
        }
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
