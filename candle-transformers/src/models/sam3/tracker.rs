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
    torch_ops::{
        interpolate::resize_bilinear2d_antialias,
        position::{build_2d_sine_position_encoding, get_1d_sine_pe},
    },
    vitdet::Sam3ViTDetTrunk,
    Config,
};

const STRICT_PORT_IN_PROGRESS: &str = "SAM3 tracker strict port in progress; legacy tracker implementation was removed. See candle-transformers/src/models/sam3/VIDEO_TRACKER_STRICT_PORT.md before implementing tracker behavior.";
const NO_OBJ_SCORE: f64 = -1024.0;

mod config;
pub use config::*;
mod maskmem_backbone;
use maskmem_backbone::TrackerSimpleMaskEncoder;
mod memory_conditioning;
mod memory_transformer;
mod model;
use memory_transformer::TrackerMemoryTransformer;
mod prompt_inputs;
use prompt_inputs::normalize_mask_prompt;
mod sam_heads;
use sam_heads::{Sam3TrackerMaskDecoder, TrackerMlp};
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
                suppress_overlapping_based_on_recent_occlusion_threshold: 0.7,
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

    mod tracker_parity {
        use super::*;

        include!("tests/tracker_parity.rs");
    }

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
