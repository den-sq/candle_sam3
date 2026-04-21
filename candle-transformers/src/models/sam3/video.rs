// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::Path;
#[cfg(test)]
use std::process::Command;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(test)]
use std::{fs, path::PathBuf};

use candle::{DType, Device, IndexOp, Result, Tensor};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

use super::{
    geometry::{EncodedPrompt, GeometryPrompt},
    image::{GroundingOutput, ImageSize},
    neck::VisualBackboneOutput,
    text::TextEncoding,
    torch_ops::{
        interpolate::resize_bilinear2d_antialias,
        masks::{binary_planes_iou, mask_to_bool_plane},
    },
    Sam3ImageModel, Sam3TrackerModel, TrackerFrameState,
};
#[cfg(test)]
#[allow(unused_imports)]
use super::{Config, Sam3TrackerConfig};

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";

mod config;
pub use config::*;
mod sources;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use sources::{frame_blob_from_rgb_image_with_filter, FrameBlob};
pub use sources::{FrameSource, VideoSource};
mod propagation;
use propagation::*;
pub use propagation::{
    ObjectFrameOutput, Sam3VideoPredictor, Sam3VideoTrackerCore, SessionPrompt, VideoFrameOutput,
    VideoOutput,
};
mod session;
pub use session::{Sam3VideoSession, SessionCacheStats, TrackedObject};
mod temporal_disambiguation;
use temporal_disambiguation::{TemporalDisambiguationFrameMetadata, TemporalDisambiguationState};
mod postprocess;
use postprocess::*;
mod prompting;
use prompting::*;
mod debug;
#[cfg(test)]
#[allow(unused_imports)]
use debug::{count_foreground_pixels, tensor_to_mask_probs_2d};
use debug::{debug_prompt_metadata, VideoDebugRecorder};

const VIDEO_DEBUG_MANIFEST_FILE: &str = "debug_manifest.json";
const VIDEO_DEBUG_MASK_THRESHOLD: f32 = 0.5;
const VIDEO_PROPAGATION_FILL_HOLE_AREA: usize = 0;
const VIDEO_PROPAGATION_HOLE_FILL_LOGIT: f32 = 0.1;
const VIDEO_PROPAGATION_SPRINKLE_REMOVE_LOGIT: f32 = -0.1;

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Tensor;
    use candle_nn::VarBuilder;
    use image::{GrayImage, ImageBuffer, ImageReader, Luma, Rgb, RgbImage};

    use crate::models::sam3::{
        Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
        Sam3TrackerConfig, SegmentationConfig, TextConfig, VisionConfig,
    };

    fn tiny_segmentation_config() -> Config {
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

    fn tiny_model(device: &Device) -> Result<Sam3ImageModel> {
        Sam3ImageModel::new(
            &tiny_segmentation_config(),
            VarBuilder::zeros(DType::F32, device),
        )
    }

    fn tiny_tracker(device: &Device) -> Result<Sam3TrackerModel> {
        let config = tiny_segmentation_config();
        let tracker_config = Sam3TrackerConfig::from_sam3_config(&config);
        Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, device))
    }

    fn sam3_test_checkpoint_path() -> Option<PathBuf> {
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

    fn tracker_config_with_reference_runtime_overrides(
        bundle: Option<&str>,
    ) -> Result<Sam3TrackerConfig> {
        let mut config = Sam3TrackerConfig::from_sam3_config(&Config::default());
        let Some(bundle) = bundle else {
            return Ok(config);
        };
        let manifest = load_reference_internal_manifest(bundle)?;
        let tracker_config = manifest["tracker_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing tracker_config".to_owned())
        })?;
        let predictor_config = manifest["predictor_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing predictor_config".to_owned())
        })?;

        if let Some(value) = tracker_config
            .get("use_memory_selection")
            .and_then(|value| value.as_bool())
        {
            config.use_memory_selection = value;
        }
        if let Some(value) = tracker_config
            .get("memory_temporal_stride_for_eval")
            .and_then(|value| value.as_u64())
        {
            config.memory_temporal_stride_for_eval = value as usize;
        }
        if let Some(value) = tracker_config
            .get("max_obj_ptrs_in_encoder")
            .and_then(|value| value.as_u64())
        {
            config.max_obj_ptrs_in_encoder = value as usize;
        }
        if let Some(value) = tracker_config
            .get("max_cond_frames_in_attn")
            .and_then(|value| value.as_u64())
        {
            config.max_cond_frames_in_attn = value as usize;
        }
        if let Some(value) = tracker_config
            .get("keep_first_cond_frame")
            .and_then(|value| value.as_bool())
        {
            config.keep_first_cond_frame = value;
        }
        if let Some(value) = tracker_config
            .get("trim_past_non_cond_mem_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.trim_past_non_cond_mem_for_eval = value;
        }
        if let Some(value) = tracker_config
            .get("offload_output_to_cpu_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.offload_output_to_cpu_for_eval = value;
        }
        if let Some(value) = tracker_config
            .get("forward_backbone_per_frame_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.forward_backbone_per_frame_for_eval = value;
        }
        if let Some(value) = predictor_config
            .get("clear_non_cond_mem_around_input")
            .and_then(|value| value.as_bool())
        {
            config.predictor.clear_non_cond_mem_around_input = value;
        }
        if let Some(value) = predictor_config
            .get("clear_non_cond_mem_for_multi_obj")
            .and_then(|value| value.as_bool())
        {
            config.predictor.clear_non_cond_mem_for_multi_obj = value;
        }
        if let Some(value) = predictor_config
            .get("always_start_from_first_ann_frame")
            .and_then(|value| value.as_bool())
        {
            config.predictor.always_start_from_first_ann_frame = value;
        }
        if let Some(value) = predictor_config
            .get("iter_use_prev_mask_pred")
            .and_then(|value| value.as_bool())
        {
            config.predictor.iter_use_prev_mask_pred = value;
        }
        if let Some(value) = predictor_config
            .get("add_all_frames_to_correct_as_cond")
            .and_then(|value| value.as_bool())
        {
            config.predictor.add_all_frames_to_correct_as_cond = value;
        }
        if let Some(value) = predictor_config
            .get("use_prev_mem_frame")
            .and_then(|value| value.as_bool())
        {
            config.predictor.use_prev_mem_frame = value;
        }
        if let Some(value) = predictor_config
            .get("use_stateless_refinement")
            .and_then(|value| value.as_bool())
        {
            config.predictor.use_stateless_refinement = value;
        }
        if let Some(value) = predictor_config
            .get("refinement_detector_cond_frame_removal_window")
            .and_then(|value| value.as_u64())
        {
            config
                .predictor
                .refinement_detector_cond_frame_removal_window = value as usize;
        }
        Ok(config)
    }

    fn load_runtime_models_from_checkpoint(
        bundle: Option<&str>,
    ) -> Result<Option<(Sam3ImageModel, Sam3TrackerModel, Device)>> {
        let Some(checkpoint_path) = sam3_test_checkpoint_path() else {
            return Ok(None);
        };
        let device = Device::Cpu;
        let config = Config::default();
        let checkpoint =
            crate::models::sam3::checkpoint::Sam3CheckpointSource::upstream_pth(checkpoint_path);
        let model =
            Sam3ImageModel::from_checkpoint_source(&config, &checkpoint, DType::F32, &device)?;
        let tracker_config = tracker_config_with_reference_runtime_overrides(bundle)?;
        let tracker = Sam3TrackerModel::new(
            &tracker_config,
            checkpoint.load_tracker_var_builder(DType::F32, &device)?,
        )?;
        Ok(Some((model, tracker, device)))
    }

    fn sam3_test_tokenizer_path() -> Option<PathBuf> {
        let checkpoint_path = sam3_test_checkpoint_path()?;
        let tokenizer = checkpoint_path.parent()?.join("tokenizer.json");
        tokenizer.exists().then_some(tokenizer)
    }

    fn reference_bundle_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../candle-examples/examples/sam3")
            .join(name)
    }

    fn reference_input_frames_dir(name: &str) -> PathBuf {
        let bundle_dir = reference_bundle_dir(name);
        let tracker_frames = bundle_dir.join("tracker_input_frames");
        if tracker_frames.exists() {
            tracker_frames
        } else {
            bundle_dir.join("frames")
        }
    }

    fn load_reference_frame_output(
        bundle: &str,
        frame_idx: usize,
    ) -> Result<(Vec<f32>, f32, PathBuf)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        let frame = frames
            .iter()
            .find(|frame| frame["frame_idx"].as_u64() == Some(frame_idx as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference video results missing frame {}",
                    frame_idx
                ))
            })?;
        let objects = frame["objects"].as_array().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} missing objects array",
                frame_idx
            ))
        })?;
        let object = &objects[0];
        let boxes = object["boxes_xyxy"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!("reference frame {} missing boxes_xyxy", frame_idx))
            })?
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let score = object["scores"]
            .as_array()
            .and_then(|scores| scores.first())
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                candle::Error::Msg(format!("reference frame {} missing score", frame_idx))
            })? as f32;
        let mask_path = object["mask_path"].as_str().ok_or_else(|| {
            candle::Error::Msg(format!("reference frame {} missing mask_path", frame_idx))
        })?;
        Ok((boxes, score, bundle_dir.join(mask_path)))
    }

    fn load_reference_object_frame_output(
        bundle: &str,
        frame_idx: usize,
        obj_id: u32,
    ) -> Result<(Vec<f32>, f32, PathBuf)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        let frame = frames
            .iter()
            .find(|frame| frame["frame_idx"].as_u64() == Some(frame_idx as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference video results missing frame {}",
                    frame_idx
                ))
            })?;
        let objects = frame["objects"].as_array().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} missing objects array",
                frame_idx
            ))
        })?;
        let object = objects
            .iter()
            .find(|object| object["obj_id"].as_u64() == Some(obj_id as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} missing obj_id {}",
                    frame_idx, obj_id
                ))
            })?;
        let boxes = object["boxes_xyxy"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} obj_id {} missing boxes_xyxy",
                    frame_idx, obj_id
                ))
            })?
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let score = object["scores"]
            .as_array()
            .and_then(|scores| scores.first())
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} obj_id {} missing score",
                    frame_idx, obj_id
                ))
            })? as f32;
        let mask_path = object["mask_path"].as_str().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} obj_id {} missing mask_path",
                frame_idx, obj_id
            ))
        })?;
        Ok((boxes, score, bundle_dir.join(mask_path)))
    }

    fn load_reference_frame_indices(bundle: &str) -> Result<Vec<usize>> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        Ok(frames
            .iter()
            .filter(|frame| {
                frame["objects"]
                    .as_array()
                    .map(|objects| !objects.is_empty())
                    .unwrap_or(false)
            })
            .filter_map(|frame| frame["frame_idx"].as_u64())
            .map(|frame_idx| frame_idx as usize)
            .collect())
    }

    fn load_reference_frame0_output(bundle: &str) -> Result<(Vec<f32>, f32, PathBuf)> {
        load_reference_frame_output(bundle, 0)
    }

    fn load_reference_box_prompt(bundle: &str) -> Result<(f32, f32, f32, f32)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let boxes = if let Some(boxes) = value["boxes_cxcywh_normalized"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
        {
            boxes.clone()
        } else {
            value["scenario"]["actions"]
                .as_array()
                .and_then(|actions| {
                    actions.iter().find(|action| {
                        action["type"].as_str() == Some("add_prompt")
                            && action["boxes_xywh"].as_array().is_some()
                    })
                })
                .and_then(|action| action["boxes_xywh"].as_array())
                .and_then(|boxes| boxes.first())
                .and_then(|first| first.as_array())
                .cloned()
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "reference bundle missing box prompt in boxes_cxcywh_normalized or scenario actions"
                            .to_owned(),
                    )
                })?
        };
        let from_scenario_xywh = value["boxes_cxcywh_normalized"]
            .as_array()
            .map(|boxes| boxes.is_empty())
            .unwrap_or(true);
        let (x0_or_cx, y0_or_cy, w, h) = (
            boxes[0].as_f64().unwrap_or(0.0) as f32,
            boxes[1].as_f64().unwrap_or(0.0) as f32,
            boxes[2].as_f64().unwrap_or(0.0) as f32,
            boxes[3].as_f64().unwrap_or(0.0) as f32,
        );
        if from_scenario_xywh {
            Ok((x0_or_cx + w * 0.5, y0_or_cy + h * 0.5, w, h))
        } else {
            Ok((x0_or_cx, y0_or_cy, w, h))
        }
    }

    fn load_reference_mask_prompt_box_xyxy(bundle: &str) -> Result<(f32, f32, f32, f32)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        let mask = actions[0]["mask"]["box_xyxy"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference mask scenario missing box_xyxy".to_owned())
        })?;
        Ok((
            mask[0].as_f64().unwrap_or(0.0) as f32,
            mask[1].as_f64().unwrap_or(0.0) as f32,
            mask[2].as_f64().unwrap_or(0.0) as f32,
            mask[3].as_f64().unwrap_or(0.0) as f32,
        ))
    }

    fn load_reference_point_prompt(bundle: &str) -> Result<(Vec<(f32, f32)>, Vec<u32>)> {
        load_reference_point_prompt_on_frame(bundle, 0)
    }

    fn load_reference_point_prompt_on_frame(
        bundle: &str,
        frame_idx: usize,
    ) -> Result<(Vec<(f32, f32)>, Vec<u32>)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        let add_prompt = actions
            .iter()
            .find(|action| {
                action["type"].as_str() == Some("add_prompt")
                    && action["frame_idx"].as_u64() == Some(frame_idx as u64)
            })
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference bundle missing add_prompt action for frame {}",
                    frame_idx
                ))
            })?;
        let points = add_prompt["points_xy_normalized"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "reference point scenario missing points_xy_normalized".to_owned(),
                )
            })?
            .iter()
            .map(|point| {
                let point = point.as_array().ok_or_else(|| {
                    candle::Error::Msg(
                        "reference point scenario contains a malformed point".to_owned(),
                    )
                })?;
                Ok((
                    point[0].as_f64().unwrap_or(0.0) as f32,
                    point[1].as_f64().unwrap_or(0.0) as f32,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let labels = add_prompt["point_labels"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg("reference point scenario missing point_labels".to_owned())
            })?
            .iter()
            .map(|value| value.as_u64().unwrap_or(0) as u32)
            .collect::<Vec<_>>();
        Ok((points, labels))
    }

    fn load_reference_internal_manifest(bundle: &str) -> Result<serde_json::Value> {
        let bundle_dir = reference_bundle_dir(bundle);
        serde_json::from_slice(&fs::read(bundle_dir.join("debug/internal_manifest.json"))?)
            .map_err(|err| candle::Error::Msg(err.to_string()))
    }

    fn apply_reference_predictor_runtime_overrides(
        predictor: &mut Sam3VideoPredictor<'_>,
        bundle: &str,
    ) -> Result<()> {
        let manifest = load_reference_internal_manifest(bundle)?;
        let predictor_config = manifest["predictor_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing predictor_config".to_owned())
        })?;
        if let Some(fill_hole_area) = predictor_config
            .get("fill_hole_area")
            .and_then(|value| value.as_u64())
        {
            predictor.video_config.fill_hole_area = fill_hole_area as usize;
        }
        if let Some(max_point_num) = predictor_config
            .get("max_point_num_in_prompt_enc")
            .and_then(|value| value.as_u64())
        {
            predictor.video_config.max_point_num_in_prompt_enc = max_point_num as usize;
        }
        if let Some(non_overlap_masks_for_output) = predictor_config
            .get("non_overlap_masks_for_output")
            .and_then(|value| value.as_bool())
        {
            predictor.video_config.non_overlap_masks_for_output = non_overlap_masks_for_output;
        }
        Ok(())
    }

    fn load_reference_internal_tensor(bundle: &str, key: &str) -> Result<Tensor> {
        use candle::safetensors::Load;

        let bundle_dir = reference_bundle_dir(bundle);
        let path = bundle_dir.join("debug/internal_fixtures.safetensors");
        let tensors =
            unsafe { candle::safetensors::MmapedSafetensors::new(&path) }.map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to mmap reference fixtures {}: {err}",
                    path.display()
                ))
            })?;
        tensors
            .get(key)
            .map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to read tensor `{key}` from reference fixtures {}: {err}",
                    path.display()
                ))
            })?
            .load(&Device::Cpu)
    }

    fn load_reference_internal_record(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
    ) -> Result<serde_json::Value> {
        let records = load_reference_internal_records(bundle, stage, frame_idx)?;
        records.into_iter().next().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference manifest missing {stage} record for frame {frame_idx}"
            ))
        })
    }

    fn load_reference_internal_records(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let manifest = load_reference_internal_manifest(bundle)?;
        Ok(manifest["records"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference manifest missing records".to_owned()))?
            .iter()
            .filter(|record| {
                record["stage"].as_str() == Some(stage)
                    && record["frame_idx"].as_u64() == Some(frame_idx as u64)
            })
            .cloned()
            .collect())
    }

    fn load_reference_internal_record_matching<F>(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
        predicate: F,
    ) -> Result<serde_json::Value>
    where
        F: Fn(&serde_json::Value) -> bool,
    {
        load_reference_internal_records(bundle, stage, frame_idx)?
            .into_iter()
            .find(predicate)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference manifest missing matching {stage} record for frame {frame_idx}"
                ))
            })
    }

    fn load_reference_internal_record_matching_last<F>(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
        predicate: F,
    ) -> Result<serde_json::Value>
    where
        F: Fn(&serde_json::Value) -> bool,
    {
        load_reference_internal_records(bundle, stage, frame_idx)?
            .into_iter()
            .rev()
            .find(predicate)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference manifest missing last matching {stage} record for frame {frame_idx}"
                ))
            })
    }

    fn load_reference_track_step_frame_output(
        bundle: &str,
        frame_idx: usize,
        video_size: ImageSize,
    ) -> Result<(Vec<f32>, f32, Tensor)> {
        let record = load_reference_internal_record(bundle, "track_step", frame_idx)?;
        let tensor_keys = record["tensor_keys"].as_object().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference track_step frame {frame_idx} missing tensor_keys"
            ))
        })?;
        let high_res_key = tensor_keys
            .get("track_step_output.pred_masks_high_res")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference track_step frame {frame_idx} missing pred_masks_high_res key"
                ))
            })?;
        let object_score_key = tensor_keys
            .get("track_step_output.object_score_logits")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference track_step frame {frame_idx} missing object_score_logits key"
                ))
            })?;
        let mask_logits = load_reference_internal_tensor(bundle, high_res_key)?;
        let resized_logits = resize_mask_logits_to_video(&mask_logits, video_size)?;
        let masks = candle_nn::ops::sigmoid(&resized_logits)?;
        let boxes = mask_to_normalized_xyxy(&masks)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let presence_score =
            candle_nn::ops::sigmoid(&load_reference_internal_tensor(bundle, object_score_key)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .next()
                .unwrap_or(0.0);
        Ok((boxes, presence_score, masks))
    }

    fn load_reference_run_single_temporal_metadata_last_per_frame(
        bundle: &str,
    ) -> Result<BTreeMap<usize, TemporalDisambiguationFrameMetadata>> {
        let manifest = load_reference_internal_manifest(bundle)?;
        let records = manifest["records"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference manifest missing records".to_owned()))?;
        let mut metadata_by_frame = BTreeMap::new();
        for record in records.iter().filter(|record| {
            record["stage"].as_str() == Some("run_single_frame_inference")
                && record["frame_idx"].as_u64().is_some()
        }) {
            let frame_idx = record["frame_idx"].as_u64().unwrap_or(0) as usize;
            let metadata = &record["metadata"];
            let read_ids = |key: &str| {
                metadata[key]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .map(|value| value.as_u64().unwrap_or(0) as u32)
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default()
            };
            metadata_by_frame.insert(
                frame_idx,
                TemporalDisambiguationFrameMetadata {
                    removed_obj_ids: read_ids("removed_obj_ids"),
                    suppressed_obj_ids: read_ids("suppressed_obj_ids"),
                    unconfirmed_obj_ids: read_ids("unconfirmed_obj_ids"),
                    matched_obj_ids: BTreeSet::new(),
                    unmatched_obj_ids: BTreeSet::new(),
                },
            );
        }
        Ok(metadata_by_frame)
    }

    fn json_usize_vec(value: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
        value[key]
            .as_array()
            .ok_or_else(|| candle::Error::Msg(format!("missing `{key}` array")))?
            .iter()
            .map(|entry| {
                entry.as_u64().map(|value| value as usize).ok_or_else(|| {
                    candle::Error::Msg(format!("malformed `{key}` entry in reference metadata"))
                })
            })
            .collect()
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

    fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        if actual.shape() != expected.shape() {
            candle::bail!(
                "shape mismatch when computing max abs diff: actual {:?}, expected {:?}",
                actual.shape().dims(),
                expected.shape().dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        actual
            .broadcast_sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()
    }

    fn binary_mask_iou(actual: &Tensor, expected_path: &Path) -> Result<f32> {
        let actual = tensor_to_mask_probs_2d(actual)?;
        let expected = image::open(expected_path)
            .map_err(|err| candle::Error::Msg(err.to_string()))?
            .to_luma8();
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (y, row) in actual.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let actual_fg = *value >= 0.5;
                let expected_fg = expected.get_pixel(x as u32, y as u32)[0] >= 128;
                if actual_fg && expected_fg {
                    intersection += 1;
                }
                if actual_fg || expected_fg {
                    union += 1;
                }
            }
        }
        Ok(if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        })
    }

    fn binary_mask_iou_tensor(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = tensor_to_mask_probs_2d(actual)?;
        let expected = tensor_to_mask_probs_2d(expected)?;
        if actual.len() != expected.len()
            || actual.first().map(Vec::len).unwrap_or(0)
                != expected.first().map(Vec::len).unwrap_or(0)
        {
            candle::bail!(
                "mask size mismatch when computing IoU from tensors: actual={}x{}, expected={}x{}",
                actual.len(),
                actual.first().map(Vec::len).unwrap_or(0),
                expected.len(),
                expected.first().map(Vec::len).unwrap_or(0)
            );
        }
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                let actual_fg = *actual_value >= 0.5;
                let expected_fg = *expected_value >= 0.5;
                if actual_fg && expected_fg {
                    intersection += 1;
                }
                if actual_fg || expected_fg {
                    union += 1;
                }
            }
        }
        Ok(if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        })
    }

    fn assert_boxes_close(actual: &[f32], expected: &[f32], atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= atol,
                "box component {idx} mismatch: actual={actual}, expected={expected}, atol={atol}"
            );
        }
    }

    fn box_mismatch_message(actual: &[f32], expected: &[f32], atol: f32) -> Option<String> {
        if actual.len() != expected.len() {
            return Some(format!(
                "box length mismatch: actual={}, expected={}",
                actual.len(),
                expected.len()
            ));
        }
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if (actual - expected).abs() > atol {
                return Some(format!(
                    "box component {idx} mismatch: actual={actual}, expected={expected}, atol={atol}"
                ));
            }
        }
        None
    }

    fn mask_tensor_to_binary_image(mask: &Tensor) -> Result<GrayImage> {
        let mask_probs = tensor_to_mask_probs_2d(mask)?;
        let height = mask_probs.len() as u32;
        let width = mask_probs.first().map(Vec::len).unwrap_or(0) as u32;
        let mut image = GrayImage::new(width, height);
        for (y, row) in mask_probs.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let pixel = if *value >= 0.5 { 255u8 } else { 0u8 };
                image.put_pixel(x as u32, y as u32, Luma([pixel]));
            }
        }
        Ok(image)
    }

    fn save_binary_mask_png(path: &Path, mask: &Tensor) -> Result<()> {
        mask_tensor_to_binary_image(mask)?
            .save(path)
            .map_err(|err| candle::Error::Msg(format!("failed to save {}: {err}", path.display())))
    }

    fn maybe_tensor_shape(tensor: Option<&Tensor>) -> Option<Vec<usize>> {
        tensor.map(|tensor| tensor.shape().dims().to_vec())
    }

    fn maybe_single_tensor_value(tensor: Option<&Tensor>) -> Result<Option<f32>> {
        match tensor {
            Some(tensor) => Ok(Some(
                tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .next()
                    .unwrap_or(0.0),
            )),
            None => Ok(None),
        }
    }

    fn dump_correction_failure_context(
        bundle: &str,
        actual8: &ObjectFrameOutput,
        actual9: &ObjectFrameOutput,
        expected_boxes8: &[f32],
        expected_score8: f32,
        expected_mask_path8: &Path,
        expected_boxes9: &[f32],
        expected_score9: f32,
        expected_mask_path9: &Path,
        frame8_state: &TrackerFrameState,
        correction_track_step: &serde_json::Value,
        correction_forward: &serde_json::Value,
        prepare_record: &serde_json::Value,
        failures: &[String],
        mask_iou8: f32,
        mask_iou9: f32,
    ) -> Result<PathBuf> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("time went backwards: {err}")))?
            .as_millis();
        let out_dir =
            PathBuf::from("/tmp/sam3_test_failures").join(format!("{}_{}", bundle, stamp));
        fs::create_dir_all(&out_dir).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create correction failure directory {}: {err}",
                out_dir.display()
            ))
        })?;

        save_binary_mask_png(&out_dir.join("actual_frame8_mask.png"), &actual8.masks)?;
        save_binary_mask_png(&out_dir.join("actual_frame9_mask.png"), &actual9.masks)?;
        fs::copy(
            expected_mask_path8,
            out_dir.join("expected_frame8_mask.png"),
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to copy {}: {err}",
                expected_mask_path8.display()
            ))
        })?;
        fs::copy(
            expected_mask_path9,
            out_dir.join("expected_frame9_mask.png"),
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to copy {}: {err}",
                expected_mask_path9.display()
            ))
        })?;

        let summary = serde_json::json!({
            "bundle": bundle,
            "failures": failures,
            "frame8": {
                "actual_boxes_xyxy": actual8.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                "expected_boxes_xyxy": expected_boxes8,
                "actual_score": actual8.score_value()?,
                "expected_score": expected_score8,
                "actual_presence_score": maybe_single_tensor_value(actual8.presence_scores.as_ref())?,
                "memory_frame_indices": actual8.memory_frame_indices,
                "mask_iou": mask_iou8,
            },
            "frame9": {
                "actual_boxes_xyxy": actual9.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                "expected_boxes_xyxy": expected_boxes9,
                "actual_score": actual9.score_value()?,
                "expected_score": expected_score9,
                "actual_presence_score": maybe_single_tensor_value(actual9.presence_scores.as_ref())?,
                "memory_frame_indices": actual9.memory_frame_indices,
                "mask_iou": mask_iou9,
            },
            "frame8_state": {
                "is_cond_frame": frame8_state.is_cond_frame,
                "maskmem_features_present": frame8_state.maskmem_features.is_some(),
                "maskmem_features_shape": maybe_tensor_shape(frame8_state.maskmem_features.as_ref()),
                "maskmem_pos_enc_present": frame8_state.maskmem_pos_enc.is_some(),
                "maskmem_pos_enc_shape": maybe_tensor_shape(frame8_state.maskmem_pos_enc.as_ref()),
                "object_score_logits": frame8_state.object_score_logits.flatten_all()?.to_vec1::<f32>()?,
            },
            "reference_internal_records": {
                "correction_track_step": correction_track_step,
                "correction_forward_sam_heads": correction_forward,
                "frame9_prepare_memory_conditioned_features": prepare_record,
            }
        });
        fs::write(
            out_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary)
                .map_err(|err| candle::Error::Msg(format!("failed to serialize summary: {err}")))?,
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write correction failure summary in {}: {err}",
                out_dir.display()
            ))
        })?;
        Ok(out_dir)
    }

    fn dump_simple_correction_failure_json(
        bundle: &str,
        phase: &str,
        details: &serde_json::Value,
    ) -> Result<PathBuf> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("time went backwards: {err}")))?
            .as_millis();
        let out_dir =
            PathBuf::from("/tmp/sam3_test_failures").join(format!("{}_{}", bundle, stamp));
        fs::create_dir_all(&out_dir).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create correction failure directory {}: {err}",
                out_dir.display()
            ))
        })?;
        fs::write(
            out_dir.join(format!("{phase}.json")),
            serde_json::to_vec_pretty(details)
                .map_err(|err| candle::Error::Msg(format!("failed to serialize summary: {err}")))?,
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write simple correction failure dump in {}: {err}",
                out_dir.display()
            ))
        })?;
        Ok(out_dir)
    }

    fn normalized_box_xyxy_to_mask_tensor(
        box_xyxy: (f32, f32, f32, f32),
        size: ImageSize,
        device: &Device,
    ) -> Result<Tensor> {
        let clamp = |value: f32| value.clamp(0.0, 1.0);
        let x0 = (clamp(box_xyxy.0) * (size.width.saturating_sub(1)) as f32).round() as usize;
        let y0 = (clamp(box_xyxy.1) * (size.height.saturating_sub(1)) as f32).round() as usize;
        let x1 = (clamp(box_xyxy.2) * (size.width.saturating_sub(1)) as f32).round() as usize;
        let y1 = (clamp(box_xyxy.3) * (size.height.saturating_sub(1)) as f32).round() as usize;
        let mut data = vec![0f32; size.height * size.width];
        if x0 <= x1 && y0 <= y1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    data[y * size.width + x] = 1.0;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, size.height, size.width), device)
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("sam3-video-tests-{}-{}", name, unique))
    }

    fn dummy_object_output(device: &Device, obj_id: u32) -> Result<ObjectFrameOutput> {
        Ok(ObjectFrameOutput {
            obj_id,
            mask_logits: Tensor::zeros((1, 4, 4), DType::F32, device)?,
            masks: Tensor::zeros((1, 4, 4), DType::F32, device)?,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, device)?,
            scores: Tensor::ones((1, 1), DType::F32, device)?,
            presence_scores: None,
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        })
    }

    fn dummy_tracker_state_for_tests(
        device: &Device,
        tracker: &Sam3TrackerModel,
    ) -> Result<TrackerFrameState> {
        Ok(TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, device)?,
            high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, device)?,
            obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: false,
        })
    }

    fn object_output_from_binary_plane(
        device: &Device,
        obj_id: u32,
        plane: &[&[u8]],
        score: f32,
        presence_score: Option<f32>,
    ) -> Result<ObjectFrameOutput> {
        let height = plane.len();
        let width = plane.first().map(|row| row.len()).unwrap_or(0);
        let data = plane
            .iter()
            .flat_map(|row| row.iter().map(|value| if *value == 0 { 0.0 } else { 1.0 }))
            .collect::<Vec<_>>();
        let binary_mask = Tensor::from_vec(data, (1, height, width), device)?;
        Ok(ObjectFrameOutput {
            obj_id,
            mask_logits: binary_mask.clone(),
            masks: binary_mask.clone(),
            boxes_xyxy: mask_to_normalized_xyxy(&binary_mask)?,
            scores: Tensor::from_vec(vec![score], (1,), device)?,
            presence_scores: presence_score
                .map(|value| Tensor::from_vec(vec![value], (1,), device))
                .transpose()?,
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        })
    }

    fn write_test_image(path: &Path, red_value: u8) -> Result<()> {
        let mut image: RgbImage = ImageBuffer::new(4, 4);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([red_value, 0, 0]);
        }
        image.save(path).map_err(|err| {
            candle::Error::Msg(format!("failed to save {}: {}", path.display(), err))
        })
    }

    fn ffmpeg_tools_available() -> bool {
        for tool in ["ffmpeg", "ffprobe"] {
            let Ok(output) = Command::new(tool).arg("-version").output() else {
                return false;
            };
            if !output.status.success() {
                return false;
            }
        }
        true
    }

    #[test]
    fn postprocess_output_applies_object_wise_non_overlap_constraints() -> Result<()> {
        let device = Device::Cpu;
        let tracker = tiny_tracker(&device)?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let model = tiny_model(&device)?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session_from_tensors(
            vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?],
            VideoSessionOptions::default(),
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.6, 0.6)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(2),
            true,
            true,
        )?;
        let results = vec![
            (
                1,
                object_output_from_binary_plane(
                    &device,
                    1,
                    &[&[1, 1, 0], &[1, 1, 0], &[0, 0, 0]],
                    0.9,
                    Some(0.9),
                )?,
                dummy_tracker_state_for_tests(&device, &tracker)?,
                Some(0.9),
            ),
            (
                2,
                object_output_from_binary_plane(
                    &device,
                    2,
                    &[&[0, 1, 1], &[0, 1, 1], &[0, 0, 0]],
                    0.4,
                    Some(0.4),
                )?,
                dummy_tracker_state_for_tests(&device, &tracker)?,
                Some(0.4),
            ),
        ];
        let mut config = VideoConfig::default();
        config.non_overlap_masks_for_output = true;
        let outputs = tracker_core.postprocess_output(
            &config,
            predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists"),
            &results,
            0.5,
            None,
            None,
            None,
        )?;
        assert_eq!(outputs.len(), 2);
        let obj1 = outputs.iter().find(|output| output.obj_id == 1).unwrap();
        let obj2 = outputs.iter().find(|output| output.obj_id == 2).unwrap();
        assert_eq!(
            count_foreground_pixels(&tensor_to_mask_probs_2d(&obj1.masks)?, 0.5),
            4
        );
        assert_eq!(
            count_foreground_pixels(&tensor_to_mask_probs_2d(&obj2.masks)?, 0.5),
            2
        );
        let obj2_mask = tensor_to_mask_probs_2d(&obj2.masks)?;
        assert!(obj2_mask[0][1] < 0.5);
        assert!(obj2_mask[1][1] < 0.5);
        Ok(())
    }

    #[test]
    fn postprocess_output_hides_unconfirmed_objects() -> Result<()> {
        let device = Device::Cpu;
        let mut tracker_config = Sam3TrackerConfig::from_sam3_config(&tiny_segmentation_config());
        tracker_config.predictor.masklet_confirmation_enable = true;
        tracker_config
            .predictor
            .masklet_confirmation_consecutive_det_thresh = 100;
        let tracker =
            Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, &device))?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let model = tiny_model(&device)?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session_from_tensors(
            vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?],
            VideoSessionOptions::default(),
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        let results = vec![(
            1,
            object_output_from_binary_plane(&device, 1, &[&[1, 1], &[1, 1]], 1.0, Some(1.0))?,
            dummy_tracker_state_for_tests(&device, &tracker)?,
            Some(1.0),
        )];
        let outputs = tracker_core.postprocess_output(
            &VideoConfig::default(),
            predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists"),
            &results,
            0.5,
            None,
            None,
            None,
        )?;
        assert!(outputs.is_empty());
        let tracked = predictor
            .sessions
            .get(&session_id)
            .and_then(|session| session.tracked_objects.get(&1))
            .expect("tracked object exists");
        assert_eq!(tracked.confirmation_consecutive_frames, 1);
        assert!(!tracked.confirmation_confirmed);
        Ok(())
    }

    #[test]
    fn postprocess_output_hides_removed_suppressed_and_unconfirmed_objects() -> Result<()> {
        let device = Device::Cpu;
        let tracker = tiny_tracker(&device)?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let model = tiny_model(&device)?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session_from_tensors(
            vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?],
            VideoSessionOptions::default(),
        )?;
        for obj_id in [1u32, 2u32, 3u32, 4u32] {
            predictor.add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: None,
                    points: Some(vec![(0.5, 0.5)]),
                    point_labels: Some(vec![1]),
                    boxes: None,
                    box_labels: None,
                },
                Some(obj_id),
                true,
                true,
            )?;
        }
        let results = vec![
            (
                1,
                object_output_from_binary_plane(&device, 1, &[&[1, 1], &[1, 1]], 0.9, Some(0.9))?,
                dummy_tracker_state_for_tests(&device, &tracker)?,
                Some(0.9),
            ),
            (
                2,
                object_output_from_binary_plane(&device, 2, &[&[1, 1], &[1, 1]], 0.8, Some(0.8))?,
                dummy_tracker_state_for_tests(&device, &tracker)?,
                Some(0.8),
            ),
            (
                3,
                object_output_from_binary_plane(&device, 3, &[&[1, 1], &[1, 1]], 0.7, Some(0.7))?,
                dummy_tracker_state_for_tests(&device, &tracker)?,
                Some(0.7),
            ),
            (
                4,
                object_output_from_binary_plane(&device, 4, &[&[1, 1], &[1, 1]], 0.6, Some(0.6))?,
                dummy_tracker_state_for_tests(&device, &tracker)?,
                Some(0.6),
            ),
        ];
        let outputs = tracker_core.postprocess_output(
            &VideoConfig::default(),
            predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists"),
            &results,
            0.5,
            Some(&[1]),
            Some(&[2]),
            Some(&[3]),
        )?;
        let actual_obj_ids = outputs
            .iter()
            .map(|output| output.obj_id)
            .collect::<Vec<_>>();
        assert_eq!(actual_obj_ids, vec![4]);
        Ok(())
    }

    fn write_test_video(video_path: &Path, red_values: &[u8]) -> Result<()> {
        let frames_dir = video_path.with_extension("frames");
        fs::create_dir_all(&frames_dir)?;
        for (idx, red_value) in red_values.iter().enumerate() {
            write_test_image(&frames_dir.join(format!("{idx}.png")), *red_value)?;
        }

        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-framerate",
                "1",
                "-start_number",
                "0",
                "-i",
            ])
            .arg(frames_dir.join("%d.png"))
            .args(["-c:v", "mpeg4", "-pix_fmt", "yuv420p"])
            .arg(video_path)
            .output()
            .map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to run ffmpeg to create {}: {}",
                    video_path.display(),
                    err
                ))
            })?;
        if !output.status.success() {
            candle::bail!(
                "ffmpeg failed to create {}: {}",
                video_path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        fs::remove_dir_all(&frames_dir)?;
        Ok(())
    }

    #[test]
    fn image_folder_frame_source_sorts_numeric_stems() -> Result<()> {
        let dir = temp_path("numeric-sort");
        fs::create_dir_all(&dir)?;
        write_test_image(&dir.join("10.png"), 200)?;
        write_test_image(&dir.join("2.png"), 100)?;
        write_test_image(&dir.join("1.png"), 50)?;

        let config = tiny_segmentation_config();
        let source = VideoSource::from_path(&dir)?;
        let mut frame_source = source.into_frame_source(&config)?;
        let frame0 = frame_source.get_frame(0, &Device::Cpu)?;
        let frame1 = frame_source.get_frame(1, &Device::Cpu)?;
        let frame2 = frame_source.get_frame(2, &Device::Cpu)?;

        let red0 = frame0.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red1 = frame1.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red2 = frame2.i((0, 0, 0))?.to_scalar::<f32>()?;
        assert!(red0 < red1);
        assert!(red1 < red2);
        frame_source.close();
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn video_file_frame_source_decodes_lazily_and_preserves_order() -> Result<()> {
        if !ffmpeg_tools_available() {
            eprintln!("skipping video decode test because ffmpeg/ffprobe are unavailable");
            return Ok(());
        }

        let dir = temp_path("video-file-source");
        fs::create_dir_all(&dir)?;
        let video_path = dir.join("clip.mp4");
        write_test_video(&video_path, &[32, 96, 160])?;

        let config = tiny_segmentation_config();
        let source = VideoSource::from_path(&video_path)?;
        let mut frame_source = source.into_frame_source(&config)?;

        assert_eq!(frame_source.frame_count(), 3);
        assert_eq!(frame_source.video_size(), ImageSize::new(4, 4));
        assert_eq!(frame_source.loaded_frame_count(), 0);

        let frame1 = frame_source.get_frame(1, &Device::Cpu)?;
        assert_eq!(frame_source.loaded_frame_count(), 1);

        frame_source.prefetch(&[0, 2])?;
        assert_eq!(frame_source.loaded_frame_count(), 3);

        let frame0 = frame_source.get_frame(0, &Device::Cpu)?;
        let frame2 = frame_source.get_frame(2, &Device::Cpu)?;
        let red0 = frame0.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red1 = frame1.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red2 = frame2.i((0, 0, 0))?.to_scalar::<f32>()?;
        assert!(
            red0 < red1,
            "expected frame 0 red {red0} < frame 1 red {red1}"
        );
        assert!(
            red1 < red2,
            "expected frame 1 red {red1} < frame 2 red {red2}"
        );

        let keep = BTreeSet::from([1usize]);
        frame_source.evict_except(&keep);
        assert_eq!(frame_source.loaded_frame_count(), 1);
        frame_source.close();
        assert_eq!(frame_source.loaded_frame_count(), 0);

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn predictor_allocates_object_ids_and_merges_prompts() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.2, 0.3)]),
                point_labels: None,
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.4, 0.5)]),
                point_labels: Some(vec![0]),
                boxes: None,
                box_labels: None,
            },
            Some(obj_id),
            false,
            true,
        )?;

        let session = predictor.sessions.get(&session_id).expect("session exists");
        let tracked = session
            .tracked_objects
            .get(&obj_id)
            .expect("tracked object exists");
        let merged = tracked.prompt_frames.get(&0).expect("prompt frame exists");
        assert_eq!(merged.text, None);
        assert_eq!(merged.points.as_ref().map(Vec::len), Some(2));
        assert_eq!(merged.point_labels.as_ref().map(Vec::len), Some(2));
        Ok(())
    }

    #[test]
    fn predictor_allows_explicit_reference_object_ids_for_new_objects() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;

        let obj_a = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.2, 0.3)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        let obj_b = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.4, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(2),
            true,
            true,
        )?;

        assert_eq!(obj_a, 1);
        assert_eq!(obj_b, 2);
        let session = predictor.sessions.get(&session_id).expect("session exists");
        assert!(session.tracked_objects.contains_key(&1));
        assert!(session.tracked_objects.contains_key(&2));
        assert_eq!(session.next_obj_id, 3);
        Ok(())
    }

    #[test]
    fn propagation_emits_directional_frames_and_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;

        let forward_options = PropagationOptions {
            direction: PropagationDirection::Forward,
            start_frame_idx: None,
            max_frame_num_to_track: None,
            output_prob_threshold: None,
        };
        let backward_options = PropagationOptions {
            direction: PropagationDirection::Backward,
            start_frame_idx: Some(1),
            max_frame_num_to_track: Some(2),
            output_prob_threshold: None,
        };
        let session = predictor
            .sessions
            .get(&session_id)
            .expect("session should exist");
        assert_eq!(
            build_processing_order(
                session,
                forward_options.direction,
                forward_options.start_frame_idx,
                forward_options.max_frame_num_to_track,
                false,
            )?,
            vec![1, 2, 3]
        );
        assert_eq!(
            build_processing_order(
                session,
                backward_options.direction,
                backward_options.start_frame_idx,
                backward_options.max_frame_num_to_track,
                false,
            )?,
            vec![1, 0]
        );

        let stats = predictor.session_cache_stats(&session_id)?;
        assert_eq!(obj_id, 0);
        assert_eq!(stats.tracked_objects, 1);
        assert!(stats.cached_feature_entries <= 2);
        Ok(())
    }

    #[test]
    fn processing_order_can_start_from_first_annotated_frame() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;
        let session = predictor
            .sessions
            .get(&session_id)
            .expect("session should exist");
        assert_eq!(
            build_processing_order(
                session,
                PropagationDirection::Forward,
                Some(3),
                Some(2),
                true
            )?,
            vec![1, 2, 3]
        );
        Ok(())
    }

    #[test]
    fn prompt_updates_preserve_current_history_and_invalidate_future_outputs() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.2, 0.3)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
            obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, &device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
            maskmem_features: Some(Tensor::zeros(
                (1, tracker.config().memory_dim, 4, 4),
                DType::F32,
                &device,
            )?),
            maskmem_pos_enc: Some(Tensor::zeros(
                (1, tracker.config().memory_dim, 4, 4),
                DType::F32,
                &device,
            )?),
            is_cond_frame: false,
        };
        let output = ObjectFrameOutput {
            obj_id,
            mask_logits: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
            masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, &device)?,
            scores: Tensor::from_vec(vec![1.0f32], (1,), &device)?,
            presence_scores: None,
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        };
        {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let tracked = session
                .tracked_objects
                .get_mut(&obj_id)
                .expect("tracked object exists");
            for frame_idx in 0..=2 {
                tracked.frame_outputs.insert(frame_idx, output.clone());
                tracked.tracker_states.insert(frame_idx, state.clone());
                session
                    .frame_outputs
                    .entry(frame_idx)
                    .or_default()
                    .insert(obj_id, output.clone());
            }
        }

        predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.4, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(obj_id),
            false,
            true,
        )?;

        let session = predictor.sessions.get(&session_id).expect("session exists");
        let tracked = session
            .tracked_objects
            .get(&obj_id)
            .expect("tracked object exists");
        assert!(tracked.frame_outputs.contains_key(&0));
        assert!(tracked.frame_outputs.contains_key(&1));
        assert!(!tracked.frame_outputs.contains_key(&2));
        assert!(tracked.tracker_states.contains_key(&0));
        assert!(tracked.tracker_states.contains_key(&1));
        assert!(!tracked.tracker_states.contains_key(&2));
        assert!(session
            .frame_outputs
            .get(&0)
            .and_then(|outputs| outputs.get(&obj_id))
            .is_some());
        assert!(session
            .frame_outputs
            .get(&1)
            .and_then(|outputs| outputs.get(&obj_id))
            .is_none());
        assert!(session
            .frame_outputs
            .get(&2)
            .and_then(|outputs| outputs.get(&obj_id))
            .is_none());
        Ok(())
    }

    #[test]
    fn clear_non_cond_mem_around_input_respects_multi_object_flag() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;

        let build_state_with_memory = |tracker: &Sam3TrackerModel| -> Result<TrackerFrameState> {
            Ok(TrackerFrameState {
                low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
                high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
                iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
                obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, &device)?,
                object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
                maskmem_features: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                maskmem_pos_enc: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                is_cond_frame: false,
            })
        };

        for clear_multi_obj in [false, true] {
            let mut tracker_config =
                Sam3TrackerConfig::from_sam3_config(&tiny_segmentation_config());
            tracker_config.predictor.clear_non_cond_mem_around_input = true;
            tracker_config.predictor.clear_non_cond_mem_for_multi_obj = clear_multi_obj;
            let tracker =
                Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, &device))?;
            let tracker_core = Sam3VideoTrackerCore::new(&tracker);
            let frames = vec![
                Tensor::zeros((3, 56, 56), DType::F32, &device)?,
                Tensor::zeros((3, 56, 56), DType::F32, &device)?,
                Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            ];
            let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
            let session_id =
                predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
            let obj_a = predictor.add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: None,
                    points: Some(vec![(0.2, 0.3)]),
                    point_labels: Some(vec![1]),
                    boxes: None,
                    box_labels: None,
                },
                None,
                true,
                true,
            )?;
            let obj_b = predictor.add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: None,
                    points: Some(vec![(0.7, 0.6)]),
                    point_labels: Some(vec![1]),
                    boxes: None,
                    box_labels: None,
                },
                None,
                true,
                true,
            )?;
            {
                let session = predictor
                    .sessions
                    .get_mut(&session_id)
                    .expect("session exists");
                for obj_id in [obj_a, obj_b] {
                    let tracked = session
                        .tracked_objects
                        .get_mut(&obj_id)
                        .expect("tracked object exists");
                    tracked
                        .tracker_states
                        .insert(1, build_state_with_memory(&tracker)?);
                }
                tracker_core.clear_non_cond_mem_around_input(session, 1);
            }
            let session = predictor.sessions.get(&session_id).expect("session exists");
            for obj_id in [obj_a, obj_b] {
                let state = session
                    .tracked_objects
                    .get(&obj_id)
                    .and_then(|object| object.tracker_states.get(&1))
                    .expect("tracker state should exist");
                assert_eq!(
                    state.maskmem_features.is_none(),
                    clear_multi_obj,
                    "clear_non_cond_mem_for_multi_obj={clear_multi_obj} should {}clear obj_id {obj_id} state",
                    if clear_multi_obj { "" } else { "not " }
                );
            }
        }
        Ok(())
    }

    #[test]
    fn image_folder_sessions_bound_loaded_frames_and_cleanup() -> Result<()> {
        let dir = temp_path("lazy-session");
        fs::create_dir_all(&dir)?;
        for (idx, value) in [32u8, 64, 96, 128, 160].iter().enumerate() {
            write_test_image(&dir.join(format!("{idx}.png")), *value)?;
        }

        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let source = VideoSource::from_path(&dir)?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session(
            source,
            VideoSessionOptions {
                prefetch_ahead: 0,
                prefetch_behind: 0,
                ..VideoSessionOptions::default()
            },
        )?;
        predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;

        let session = predictor
            .sessions
            .get_mut(&session_id)
            .expect("session should exist");
        session.prefetch_for_frame(1, PropagationDirection::Forward)?;
        let _ = session.get_frame(1, &device)?;
        session.evict_for_frame(1, PropagationDirection::Forward);
        assert!(
            session.frame_source.loaded_frame_count() <= 2,
            "expected lazy source to keep prompt/current frames only, got {} loaded frames",
            session.frame_source.loaded_frame_count()
        );
        session.close();
        assert_eq!(session.frame_source.loaded_frame_count(), 0);

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn remove_object_clears_cached_outputs() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;

        let cached = dummy_object_output(&device, obj_id)?;
        predictor
            .sessions
            .get_mut(&session_id)
            .expect("session exists")
            .tracked_objects
            .get_mut(&obj_id)
            .expect("tracked object exists")
            .frame_outputs
            .insert(0, cached.clone());
        predictor
            .sessions
            .get_mut(&session_id)
            .expect("session exists")
            .frame_outputs
            .entry(0)
            .or_default()
            .insert(obj_id, cached);
        predictor.remove_object(&session_id, obj_id)?;

        let session = predictor.sessions.get(&session_id).expect("session exists");
        assert!(!session.tracked_objects.contains_key(&obj_id));
        assert!(session
            .frame_outputs
            .values()
            .all(|objects| objects.is_empty()));
        Ok(())
    }

    #[test]
    fn text_prompts_require_tokenizer_path() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let err = predictor
            .add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: Some("person".to_owned()),
                    points: None,
                    point_labels: None,
                    boxes: None,
                    box_labels: None,
                },
                None,
                true,
                true,
            )
            .expect_err("text prompt should require tokenizer");
        assert!(
            err.to_string().contains("tokenizer"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[test]
    fn probability_masks_use_half_threshold_for_box_extraction() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(
            vec![
                0.1f32, 0.1, 0.1, 0.1, //
                0.1, 0.9, 0.9, 0.1, //
                0.1, 0.9, 0.9, 0.1, //
                0.1, 0.1, 0.1, 0.1, //
            ],
            (1, 1, 4, 4),
            &device,
        )?;
        let boxes = mask_to_normalized_xyxy(&mask)?;
        assert_eq!(boxes.to_vec2::<f32>()?, vec![vec![0.25, 0.25, 0.75, 0.75]]);
        Ok(())
    }

    #[test]
    fn tracker_outputs_are_resized_to_video_space_and_scores_are_flattened() -> Result<()> {
        let device = Device::Cpu;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?,
            high_res_masks: Tensor::from_vec(
                vec![
                    -5.0f32, -5.0, -5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, -5.0, -5.0, -5.0, //
                ],
                (1, 1, 4, 4),
                &device,
            )?,
            iou_scores: Tensor::from_vec(vec![0.25f32], (1, 1), &device)?,
            obj_ptr: Tensor::zeros((1, 8), DType::F32, &device)?,
            object_score_logits: Tensor::from_vec(vec![2.0f32], (1, 1), &device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: true,
        };
        let output = tracker_state_to_object_output(
            7,
            &state,
            None,
            Some(0),
            vec![0, 1, 2],
            None,
            true,
            false,
            ImageSize::new(2, 6),
        )?;
        assert_eq!(output.mask_logits.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.masks.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.scores.to_vec1::<f32>()?, vec![0.25]);
        assert_eq!(output.boxes_xyxy.dims(), &[1, 4]);
        Ok(())
    }

    #[test]
    fn tracker_outputs_can_use_persistent_display_scores() -> Result<()> {
        let device = Device::Cpu;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?,
            iou_scores: Tensor::from_vec(vec![0.25f32], (1, 1), &device)?,
            obj_ptr: Tensor::zeros((1, 8), DType::F32, &device)?,
            object_score_logits: Tensor::from_vec(vec![2.0f32], (1, 1), &device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: true,
        };
        let output = tracker_state_to_object_output(
            7,
            &state,
            Some(0.9),
            Some(0),
            vec![0],
            None,
            true,
            false,
            ImageSize::new(2, 2),
        )?;
        assert_eq!(output.scores.to_vec1::<f32>()?, vec![0.9]);
        Ok(())
    }

    #[test]
    fn detector_seed_outputs_are_resized_to_video_space_from_masks() -> Result<()> {
        let device = Device::Cpu;
        let grounding = GroundingOutput {
            mask_logits: Tensor::from_vec(
                vec![
                    -5.0f32, -5.0, -5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, -5.0, -5.0, -5.0, //
                ],
                (1, 4, 4),
                &device,
            )?,
            masks: Tensor::zeros((1, 4, 4), DType::F32, &device)?,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, &device)?,
            scores: Tensor::from_vec(vec![0.75f32], (1, 1), &device)?,
            presence_scores: Some(Tensor::from_vec(vec![1.5f32], (1, 1), &device)?),
        };
        let output = grounding_to_object_output(
            9,
            &grounding,
            Some(0),
            vec![],
            None,
            true,
            false,
            ImageSize::new(2, 6),
        )?;
        assert_eq!(output.mask_logits.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.masks.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.scores.to_vec1::<f32>()?, vec![0.75]);
        assert_eq!(
            output
                .presence_scores
                .as_ref()
                .expect("presence scores should be preserved")
                .to_vec1::<f32>()?,
            vec![1.5]
        );
        assert_ne!(
            output.boxes_xyxy.to_vec2::<f32>()?,
            vec![vec![0.0, 0.0, 0.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn propagation_mask_postprocess_matches_upstream_fill_and_sprinkle_rules() -> Result<()> {
        let device = Device::Cpu;
        let mask_logits = Tensor::from_vec(
            vec![
                -1.0f32, -1.0, -1.0, -1.0, -1.0, //
                -1.0, 1.0, 1.0, 1.0, -1.0, //
                -1.0, 1.0, -1.0, 1.0, -1.0, //
                -1.0, 1.0, 1.0, 1.0, -1.0, //
                1.0, -1.0, -1.0, -1.0, -1.0, //
            ],
            (1, 1, 5, 5),
            &device,
        )?;
        let postprocessed = postprocess_low_res_mask_logits_for_video(&mask_logits, 1)?;
        assert_eq!(
            postprocessed.i((0, 0))?.to_vec2::<f32>()?,
            vec![
                vec![-1.0, -1.0, -1.0, -1.0, -1.0],
                vec![-1.0, 1.0, 1.0, 1.0, -1.0],
                vec![-1.0, 1.0, 0.1, 1.0, -1.0],
                vec![-1.0, 1.0, 1.0, 1.0, -1.0],
                vec![1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        );
        Ok(())
    }

    #[test]
    #[ignore = "diagnostic for direct-tracker visual feature parity"]
    fn video_frame0_visual_features_match_single_click_point_fixture_bundle() -> Result<()> {
        let Some((model, tracker, device)) =
            load_runtime_models_from_checkpoint(Some("reference_video_point_debug_single_click"))?
        else {
            return Ok(());
        };
        let bundle = "reference_video_point_debug_single_click";
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let (preprocessed_image, visual, raw_visual) = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let image = session.get_frame(0, &device)?.unsqueeze(0)?;
            let raw_visual = session.get_visual_features(&model, &device, 0)?;
            let visual = tracker_visual_output(&raw_visual);
            (image, visual, raw_visual)
        };

        let manifest = load_reference_internal_manifest(bundle)?;
        let records = manifest["records"].as_array().ok_or_else(|| {
            candle::Error::Msg("single-click manifest missing records".to_owned())
        })?;
        let get_image_feature_record = records.iter().find(|record| {
            record["frame_idx"].as_u64() == Some(0)
                && record["stage"].as_str() == Some("get_image_feature")
        });
        let (expected_image, expected_backbone, expected_high_res_0, expected_high_res_1) =
            if let Some(record) = get_image_feature_record {
                let keys = record["tensor_keys"].as_object().ok_or_else(|| {
                    candle::Error::Msg(
                        "single-click get_image_feature record missing tensor keys".to_owned(),
                    )
                })?;
                (
                    load_reference_internal_tensor(bundle, keys["image"].as_str().unwrap())?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_out.backbone_fpn.2"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_out.backbone_fpn.0"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_out.backbone_fpn.1"].as_str().unwrap(),
                    )?,
                )
            } else {
                let record = records
                    .iter()
                    .find(|record| {
                        record["frame_idx"].as_u64() == Some(0)
                            && record["stage"].as_str() == Some("forward_sam_heads")
                    })
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "missing single-click get_image_feature/forward_sam_heads record"
                                .to_owned(),
                        )
                    })?;
                let keys = record["tensor_keys"].as_object().ok_or_else(|| {
                    candle::Error::Msg(
                        "single-click forward_sam_heads record missing tensor keys".to_owned(),
                    )
                })?;
                (
                    Tensor::zeros(preprocessed_image.shape(), DType::F32, &Device::Cpu)?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_features"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["high_res_features.0"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["high_res_features.1"].as_str().unwrap(),
                    )?,
                )
            };

        if get_image_feature_record.is_some() {
            assert_tensor_close(
                "single-click preprocessed image",
                &preprocessed_image.to_device(&Device::Cpu)?,
                &expected_image,
                1e-4,
            )?;
        }

        let tracker_backbone = visual.backbone_fpn[2].to_device(&Device::Cpu)?;
        let tracker_backbone_diff = tensor_max_abs_diff(&tracker_backbone, &expected_backbone)?;
        if tracker_backbone_diff > 1e-3 {
            let primary_backbone_diff = tensor_max_abs_diff(
                &raw_visual.backbone_fpn[2].to_device(&Device::Cpu)?,
                &expected_backbone,
            )?;
            let sam2_backbone_diff = raw_visual
                .sam2_backbone_fpn
                .as_ref()
                .map(|levels| {
                    tensor_max_abs_diff(&levels[2].to_device(&Device::Cpu)?, &expected_backbone)
                })
                .transpose()?;
            candle::bail!(
                "single-click tracker backbone feature mismatch: tracker_diff={tracker_backbone_diff:.6}, primary_diff={primary_backbone_diff:.6}, sam2_diff={:.6}",
                sam2_backbone_diff.unwrap_or(primary_backbone_diff),
            );
        }
        let projected_high_res =
            tracker.prepare_high_res_features_for_test(&visual.backbone_fpn[..2])?;
        assert_tensor_close(
            "single-click projected high_res feature 0",
            &projected_high_res[0].to_device(&Device::Cpu)?,
            &expected_high_res_0,
            1e-3,
        )?;
        assert_tensor_close(
            "single-click projected high_res feature 1",
            &projected_high_res[1].to_device(&Device::Cpu)?,
            &expected_high_res_1,
            1e-3,
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "diagnostic for video preprocessing filter parity"]
    fn video_frame0_preprocess_filter_diagnostics_against_single_click_fixture_bundle() -> Result<()>
    {
        let bundle = "reference_video_point_debug_single_click";
        let manifest = load_reference_internal_manifest(bundle)?;
        let records = manifest["records"].as_array().ok_or_else(|| {
            candle::Error::Msg("single-click manifest missing records".to_owned())
        })?;
        let record = records
            .iter()
            .find(|record| {
                record["frame_idx"].as_u64() == Some(0)
                    && record["stage"].as_str() == Some("get_image_feature")
            })
            .ok_or_else(|| {
                candle::Error::Msg("missing single-click get_image_feature record".to_owned())
            })?;
        let keys = record["tensor_keys"].as_object().ok_or_else(|| {
            candle::Error::Msg(
                "single-click get_image_feature record missing tensor keys".to_owned(),
            )
        })?;
        let expected_image =
            load_reference_internal_tensor(bundle, keys["image"].as_str().unwrap())?;

        let frame_path = reference_input_frames_dir(bundle).join("000000.jpg");
        let image = ImageReader::open(&frame_path)?
            .decode()
            .map_err(candle::Error::wrap)?
            .to_rgb8();
        let (width, height) = image.dimensions();

        let filters = [
            ("Nearest", image::imageops::FilterType::Nearest),
            ("Triangle", image::imageops::FilterType::Triangle),
            ("CatmullRom", image::imageops::FilterType::CatmullRom),
            ("Gaussian", image::imageops::FilterType::Gaussian),
            ("Lanczos3", image::imageops::FilterType::Lanczos3),
        ];
        let mut lines = Vec::new();
        for (label, filter) in filters {
            let frame = frame_blob_from_rgb_image_with_filter(
                image.clone(),
                1008,
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                ImageSize::new(height as usize, width as usize),
                &frame_path.display().to_string(),
                filter,
            )?;
            let actual = Tensor::from_vec(
                frame.data,
                (1, 3, frame.frame_size.height, frame.frame_size.width),
                &Device::Cpu,
            )?;
            let diff = tensor_max_abs_diff(&actual, &expected_image)?;
            lines.push(format!("{label}: {diff:.6}"));
        }
        candle::bail!(
            "single-click preprocess filter diagnostics -> {}",
            lines.join(", ")
        );
    }

    mod video_parity {
        use super::*;

        include!("tests/video_parity.rs");
    }

    fn memory_frame_indices_trim_to_configured_window() {
        assert_eq!(VideoConfig::default().memory_frame_count, 6);
        assert_eq!(
            trim_memory_frame_indices(vec![0, 1, 2, 3, 4, 5, 6], 6),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(trim_memory_frame_indices(vec![3, 4], 6), vec![3, 4]);
    }

    #[test]
    fn video_trim_past_non_cond_memory_clears_only_old_maskmem() -> Result<()> {
        let device = Device::Cpu;
        let mut tracker_config = Sam3TrackerConfig::from_sam3_config(&tiny_segmentation_config());
        tracker_config.predictor.trim_past_non_cond_mem_for_eval = true;
        tracker_config.use_memory_selection = true;
        tracker_config.max_obj_ptrs_in_encoder = 1;
        let tracker =
            Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, &device))?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let model = tiny_model(&device)?;
        let frames = vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let state_with_memory = |is_cond_frame: bool| -> Result<TrackerFrameState> {
            Ok(TrackerFrameState {
                low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
                high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
                iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
                obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, &device)?,
                object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
                maskmem_features: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                maskmem_pos_enc: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                is_cond_frame,
            })
        };
        {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let tracked = session
                .tracked_objects
                .get_mut(&obj_id)
                .expect("tracked object exists");
            tracked.tracker_states.insert(4, state_with_memory(false)?);
            tracked.tracker_states.insert(17, state_with_memory(false)?);
            tracked.tracker_states.insert(22, state_with_memory(false)?);
            tracked.tracker_states.insert(23, state_with_memory(false)?);
            tracked.tracker_states.insert(10, state_with_memory(true)?);
            let snapshot = tracked.clone();
            tracker_core.trim_past_non_cond_memory(
                session,
                &snapshot,
                24,
                PropagationDirection::Forward,
            );
        }
        let tracked = predictor
            .sessions
            .get(&session_id)
            .and_then(|session| session.tracked_objects.get(&obj_id))
            .expect("tracked object exists after trim");
        assert!(tracked
            .tracker_states
            .get(&4)
            .expect("far-old non-cond frame exists")
            .maskmem_features
            .is_none());
        assert!(tracked
            .tracker_states
            .get(&4)
            .expect("far-old non-cond frame exists")
            .maskmem_pos_enc
            .is_none());
        assert!(tracked
            .tracker_states
            .get(&17)
            .expect("memory-window non-cond frame exists")
            .maskmem_features
            .is_none());
        assert!(tracked
            .tracker_states
            .get(&10)
            .expect("conditioning frame exists")
            .maskmem_features
            .is_some());
        assert!(tracked
            .tracker_states
            .get(&23)
            .expect("recent non-cond frame exists")
            .maskmem_features
            .is_some());
        assert_eq!(
            tracked
                .tracker_states
                .get(&4)
                .expect("far-old non-cond frame exists")
                .obj_ptr
                .shape()
                .dims(),
            &[1, tracker.config().hidden_dim]
        );
        Ok(())
    }

    #[test]
    fn debug_capture_writes_seed_and_first_propagation_records() -> Result<()> {
        let device = Device::Cpu;
        let debug_root = temp_path("debug-capture");
        let mut recorder = VideoDebugRecorder::new(
            "session_0",
            VideoDebugConfig {
                enabled: true,
                capture_obj_ids: Vec::new(),
                capture_frame_indices: vec![0, 1],
                capture_first_propagated_only: true,
                output_root: Some(debug_root.clone()),
            },
        )?
        .expect("debug recorder should be created");
        let mut object = TrackedObject::new(0, 0);
        object.prompt_frames.insert(
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.25, 0.25)]),
                box_labels: Some(vec![1]),
            },
        );
        let mask_logits = Tensor::from_vec(
            vec![
                -10.0f32, -10.0, //
                -10.0, 10.0, //
            ],
            (1, 1, 2, 2),
            &device,
        )?;
        let masks = candle_nn::ops::sigmoid(&mask_logits)?;
        let output = ObjectFrameOutput {
            obj_id: 0,
            mask_logits: mask_logits.clone(),
            masks: masks.clone(),
            boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
            scores: Tensor::from_vec(vec![0.9f32], (1,), &device)?,
            presence_scores: Some(Tensor::from_vec(vec![0.8f32], (1,), &device)?),
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        };
        let state = TrackerFrameState {
            low_res_masks: mask_logits.clone(),
            high_res_masks: mask_logits.clone(),
            iou_scores: Tensor::from_vec(vec![0.9f32], (1, 1), &device)?,
            obj_ptr: Tensor::zeros((1, 8), DType::F32, &device)?,
            object_score_logits: Tensor::from_vec(vec![1.0f32], (1, 1), &device)?,
            maskmem_features: Some(Tensor::zeros((1, 8, 1, 1), DType::F32, &device)?),
            maskmem_pos_enc: Some(Tensor::zeros((1, 8, 1, 1), DType::F32, &device)?),
            is_cond_frame: true,
        };
        let prompt_metadata =
            debug_prompt_metadata(object.prompt_frames.get(&0).expect("prompt exists"), true)?;
        recorder.record_detector_grounding(
            &object,
            0,
            PropagationDirection::Forward,
            prompt_metadata.clone(),
            &output,
        )?;
        recorder.record_tracker_seed(
            &object,
            0,
            PropagationDirection::Forward,
            prompt_metadata,
            &output,
            &state,
        )?;
        let mut history = BTreeMap::new();
        history.insert(0, state);
        recorder.record_first_propagation(
            &object,
            1,
            PropagationDirection::Forward,
            Some(0),
            &ObjectFrameOutput {
                prompt_frame_idx: Some(0),
                memory_frame_indices: vec![0],
                reused_previous_output: true,
                ..output
            },
            &history,
            &[0],
            &[0],
        )?;

        let manifest: serde_json::Value = serde_json::from_str(&fs::read_to_string(
            debug_root.join(VIDEO_DEBUG_MANIFEST_FILE),
        )?)
        .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let records = manifest["records"]
            .as_array()
            .expect("debug manifest records should be an array");
        assert!(records
            .iter()
            .any(|record| record["stage"].as_str() == Some("detector_grounding")));
        assert!(records
            .iter()
            .any(|record| record["stage"].as_str() == Some("tracker_seed")));
        let propagation_input = records
            .iter()
            .find(|record| record["stage"].as_str() == Some("propagation_input"))
            .expect("propagation input should be captured");
        assert_eq!(propagation_input["frame_idx"].as_u64(), Some(1));
        let propagation_input = propagation_input["propagation_input"]
            .as_object()
            .expect("propagation input summary should be present");
        assert_eq!(
            propagation_input["history_frame_order"],
            serde_json::json!([0])
        );
        assert_eq!(
            propagation_input["chosen_prompt_frame_indices"],
            serde_json::json!([0])
        );
        assert_eq!(
            propagation_input["chosen_memory_frame_indices"],
            serde_json::json!([0])
        );
        assert_eq!(
            propagation_input["history_frames"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(
            propagation_input["history_frames"][0]["is_cond_frame"].as_bool(),
            Some(true)
        );
        let detector = records
            .iter()
            .find(|record| record["stage"].as_str() == Some("detector_grounding"))
            .and_then(|record| record["observable"].as_object())
            .expect("detector observable should be present");
        let detector_mask = image::open(
            debug_root.join(
                detector
                    .get("mask_path")
                    .and_then(|value| value.as_str())
                    .expect("detector mask path should be present"),
            ),
        )
        .map_err(|err| candle::Error::Msg(err.to_string()))?
        .to_luma8();
        assert!(detector_mask
            .pixels()
            .all(|pixel| matches!(pixel[0], 0 | 255)));
        Ok(())
    }

    #[test]
    fn disabled_debug_capture_writes_no_artifacts_and_keeps_outputs_stable() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let prompt = SessionPrompt {
            text: None,
            points: None,
            point_labels: None,
            boxes: Some(vec![(0.5, 0.5, 0.25, 0.25)]),
            box_labels: Some(vec![1]),
        };

        let mut baseline = Sam3VideoPredictor::new(&model, &tracker, &device);
        let baseline_session = baseline.start_session_from_tensors(
            frames.iter().cloned().collect(),
            VideoSessionOptions::default(),
        )?;
        let baseline_obj_id =
            baseline.add_prompt(&baseline_session, 0, prompt.clone(), None, true, true)?;

        let debug_root = temp_path("debug-disabled");
        let mut debug_predictor = Sam3VideoPredictor::new(&model, &tracker, &device)
            .with_debug_config(VideoDebugConfig {
                enabled: false,
                capture_obj_ids: Vec::new(),
                capture_frame_indices: vec![0, 1],
                capture_first_propagated_only: true,
                output_root: Some(debug_root.clone()),
            });
        let debug_session =
            debug_predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let debug_obj_id =
            debug_predictor.add_prompt(&debug_session, 0, prompt, None, true, true)?;

        let baseline_prompt = baseline
            .sessions
            .get(&baseline_session)
            .and_then(|session| session.tracked_objects.get(&baseline_obj_id))
            .and_then(|object| object.prompt_frames.get(&0))
            .expect("baseline prompt should exist");
        let debug_prompt = debug_predictor
            .sessions
            .get(&debug_session)
            .and_then(|session| session.tracked_objects.get(&debug_obj_id))
            .and_then(|object| object.prompt_frames.get(&0))
            .expect("debug prompt should exist");

        assert_eq!(baseline_obj_id, debug_obj_id);
        assert_eq!(baseline_prompt.text, debug_prompt.text);
        assert_eq!(baseline_prompt.points, debug_prompt.points);
        assert_eq!(baseline_prompt.point_labels, debug_prompt.point_labels);
        assert_eq!(baseline_prompt.boxes, debug_prompt.boxes);
        assert_eq!(baseline_prompt.box_labels, debug_prompt.box_labels);
        assert_eq!(
            baseline
                .session_cache_stats(&baseline_session)?
                .tracked_objects,
            debug_predictor
                .session_cache_stats(&debug_session)?
                .tracked_objects
        );

        baseline.close_session(&baseline_session)?;
        debug_predictor.close_session(&debug_session)?;
        assert!(!debug_root.join(VIDEO_DEBUG_MANIFEST_FILE).exists());
        assert!(!debug_root.exists());
        Ok(())
    }
}
