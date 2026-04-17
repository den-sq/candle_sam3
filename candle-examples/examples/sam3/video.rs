// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use candle::Device;
use candle::{IndexOp, Tensor};
use candle_transformers::models::sam3;
use image::{ImageReader, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

const VIDEO_REFERENCE_METADATA_FILE: &str = "reference.json";
const VIDEO_RESULTS_FILE: &str = "video_results.json";
const VIDEO_FRAMES_DIR: &str = "frames";
const VIDEO_MASKS_DIR: &str = "masks";
const VIDEO_MASKED_FRAMES_DIR: &str = "masked_frames";
const VIDEO_DEBUG_DIR: &str = "debug";
const VIDEO_DEBUG_MANIFEST_FILE: &str = "debug_manifest.json";
const VIDEO_DEBUG_COMPARE_FILE: &str = "debug_compare.json";
const MASK_COLOR: [u8; 3] = [56, 201, 84];
const MASK_THRESHOLD: f32 = 0.5;

pub struct VideoMode {
    pub video_path: String,
    pub tokenizer_path: Option<String>,
    pub prompt_text: Option<String>,
    pub points: Vec<(f32, f32)>,
    pub point_labels: Vec<u32>,
    pub boxes: Vec<(f32, f32, f32, f32)>,
    pub box_labels: Vec<u32>,
    pub frame_stride: usize,
    pub prefetch_ahead: usize,
    pub prefetch_behind: usize,
    pub max_feature_cache_entries: usize,
    pub offload_state_to_cpu: bool,
    pub debug_bundle: bool,
    pub debug_obj_ids: Vec<u32>,
    pub debug_frame_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoExportMetadata {
    #[serde(default = "default_bundle_version")]
    bundle_version: usize,
    mode: String,
    source_path: String,
    source_kind: String,
    session_frame_count: usize,
    exported_frame_count: usize,
    frame_stride: usize,
    tokenizer_path: Option<String>,
    prompt_text: Option<String>,
    points_xy_normalized: Vec<Vec<f32>>,
    point_labels: Vec<u32>,
    boxes_cxcywh_normalized: Vec<Vec<f32>>,
    box_labels: Vec<u32>,
    frames_dir: String,
    masks_dir: String,
    masked_frames_dir: String,
    results_path: String,
    #[serde(default)]
    debug_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoFrameRecord {
    frame_idx: usize,
    frame_path: String,
    objects: Vec<VideoObjectRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoObjectRecord {
    obj_id: u32,
    scores: Vec<f32>,
    presence_scores: Option<Vec<f32>>,
    boxes_xyxy: Vec<Vec<f32>>,
    mask_path: Option<String>,
    masked_frame_path: Option<String>,
    prompt_frame_idx: Option<usize>,
    memory_frame_indices: Vec<usize>,
    text_prompt: Option<String>,
    used_explicit_geometry: bool,
    reused_previous_output: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugManifest {
    bundle_version: usize,
    mode: String,
    source: String,
    session_id: String,
    internal_tracker_state_available: bool,
    #[serde(default)]
    capture_obj_ids: Vec<u32>,
    #[serde(default)]
    capture_frame_indices: Vec<usize>,
    capture_first_propagated_only: bool,
    records: Vec<VideoDebugRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugRecord {
    stage: String,
    obj_id: u32,
    frame_idx: usize,
    prompt_frame_idx: Option<usize>,
    prompt_metadata: Option<VideoDebugPromptMetadata>,
    observable: Option<VideoDebugObservableSummary>,
    tracker_state: Option<VideoDebugTrackerStateSummary>,
    propagation_input: Option<VideoDebugPropagationInputSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugPromptMetadata {
    text_prompt: Option<String>,
    used_visual_text_prompt: bool,
    normalized_points_xy: Vec<Vec<f32>>,
    point_labels: Vec<u32>,
    normalized_boxes_cxcywh: Vec<Vec<f32>>,
    box_labels: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugObservableSummary {
    mask_path: Option<String>,
    mask_threshold: f32,
    foreground_pixel_count: usize,
    mask_area_ratio: f32,
    boxes_xyxy: Vec<Vec<f32>>,
    scores: Vec<f32>,
    presence_scores: Option<Vec<f32>>,
    mask_logits_stats: TensorDebugSummary,
    mask_prob_stats: TensorDebugSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugTrackerStateSummary {
    is_cond_frame: bool,
    low_res_masks_stats: TensorDebugSummary,
    high_res_masks_stats: TensorDebugSummary,
    iou_scores_stats: TensorDebugSummary,
    object_score_logits_stats: TensorDebugSummary,
    obj_ptr_stats: TensorDebugSummary,
    maskmem_features_stats: Option<TensorDebugSummary>,
    maskmem_pos_enc_stats: Option<TensorDebugSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugPropagationInputSummary {
    history_frames: Vec<VideoDebugHistoryFrameSummary>,
    history_frame_order: Vec<usize>,
    chosen_prompt_frame_indices: Vec<usize>,
    chosen_memory_frame_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugHistoryFrameSummary {
    frame_idx: usize,
    is_cond_frame: bool,
    low_res_masks_stats: TensorDebugSummary,
    high_res_masks_stats: TensorDebugSummary,
    obj_ptr_stats: TensorDebugSummary,
    maskmem_features_stats: Option<TensorDebugSummary>,
    maskmem_pos_enc_stats: Option<TensorDebugSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorDebugSummary {
    shape: Vec<usize>,
    dtype: String,
    min: f32,
    max: f32,
    mean: f32,
    l2_norm: f32,
    foreground_pixel_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoDebugComparisonReport {
    reference_bundle: String,
    candle_output_dir: String,
    seed_frame_prompt_vs_detector: Option<VideoDebugMetricSummary>,
    seed_frame_detector_vs_tracker_seed: Option<VideoDebugMetricSummary>,
    first_propagated_vs_reference: Option<VideoDebugMetricSummary>,
    candle_area_growth_ratio: Option<f32>,
    reference_area_growth_ratio: Option<f32>,
    verdict: String,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoDebugMetricSummary {
    mask_iou_threshold_0_5: Option<f32>,
    box_iou: Option<f32>,
    mean_box_abs_diff: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoReferenceComparisonReport {
    reference_bundle: String,
    candle_output_dir: String,
    reference_frame_count: usize,
    candle_frame_count: usize,
    compared_frame_count: usize,
    all_reference_frames_present: bool,
    all_reference_objects_present: bool,
    mean_score_abs_diff: Option<f32>,
    mean_box_l1_abs_diff: Option<f32>,
    mean_box_iou: Option<f32>,
    mean_mask_abs_diff: Option<f32>,
    mean_mask_iou_threshold_0_5: Option<f32>,
    mean_masked_frame_abs_diff: Option<f32>,
    mean_masked_frame_rmse: Option<f32>,
    frame_reports: Vec<VideoFrameComparisonReport>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoFrameComparisonReport {
    frame_idx: usize,
    reference_object_count: usize,
    candle_object_count: usize,
    all_objects_present: bool,
    object_reports: Vec<VideoObjectComparisonReport>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoObjectComparisonReport {
    obj_id: u32,
    reference_score: Option<f32>,
    candle_score: Option<f32>,
    score_abs_diff: Option<f32>,
    reference_box_xyxy: Option<Vec<f32>>,
    candle_box_xyxy: Option<Vec<f32>>,
    box_l1_mean_abs_diff: Option<f32>,
    box_iou: Option<f32>,
    mask_mean_abs_diff: Option<f32>,
    mask_iou_threshold_0_5: Option<f32>,
    masked_frame_mean_abs_diff: Option<f32>,
    masked_frame_rmse: Option<f32>,
    notes: Vec<String>,
}

enum ExportFrameSource {
    ImagePaths(Vec<PathBuf>),
    VideoFile(PathBuf),
}

impl ExportFrameSource {
    fn new(source_path: &Path) -> Result<Self> {
        if source_path.is_dir() {
            return Ok(Self::ImagePaths(sorted_image_paths(source_path)?));
        }
        let ext = source_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());
        match ext.as_deref() {
            Some("jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp") => {
                Ok(Self::ImagePaths(vec![source_path.to_path_buf()]))
            }
            Some("mp4" | "avi" | "mov" | "mkv" | "webm") => {
                Ok(Self::VideoFile(source_path.to_path_buf()))
            }
            _ => bail!("unsupported video export source {}", source_path.display()),
        }
    }

    fn source_kind(&self) -> &'static str {
        match self {
            Self::ImagePaths(paths) if paths.len() == 1 => "image_file",
            Self::ImagePaths(_) => "image_folder",
            Self::VideoFile(_) => "video_file",
        }
    }

    fn load_rgba(&self, frame_idx: usize) -> Result<RgbaImage> {
        match self {
            Self::ImagePaths(paths) => {
                let image_path = paths.get(frame_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "frame_idx {} out of bounds for {} image frames",
                        frame_idx,
                        paths.len()
                    )
                })?;
                Ok(ImageReader::open(image_path)?
                    .decode()
                    .map_err(anyhow::Error::from)?
                    .to_rgba8())
            }
            Self::VideoFile(video_path) => decode_video_frame_rgba(video_path, frame_idx),
        }
    }
}

fn default_bundle_version() -> usize {
    1
}

pub fn run_video_prediction(
    model: &sam3::Sam3ImageModel,
    tracker: &sam3::Sam3TrackerModel,
    video_mode: &VideoMode,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    println!("Starting video prediction for: {}", video_mode.video_path);

    let source_path = PathBuf::from(&video_mode.video_path);
    let source = sam3::VideoSource::from_path(&video_mode.video_path)?;
    let session_options = sam3::VideoSessionOptions {
        tokenizer_path: video_mode.tokenizer_path.as_ref().map(PathBuf::from),
        offload_frames_to_cpu: true,
        offload_state_to_cpu: video_mode.offload_state_to_cpu,
        prefetch_ahead: video_mode.prefetch_ahead,
        prefetch_behind: video_mode.prefetch_behind,
        max_feature_cache_entries: video_mode.max_feature_cache_entries,
    };
    let debug_root = output_dir.join(VIDEO_DEBUG_DIR);
    if video_mode.debug_bundle {
        clear_output_dir(&debug_root)?;
    }

    let mut predictor = sam3::Sam3VideoPredictor::new(model, tracker, device).with_debug_config(
        sam3::VideoDebugConfig {
            enabled: video_mode.debug_bundle,
            capture_obj_ids: video_mode.debug_obj_ids.clone(),
            capture_frame_indices: video_mode.debug_frame_indices.clone(),
            capture_first_propagated_only: true,
            output_root: video_mode.debug_bundle.then_some(debug_root.clone()),
        },
    );
    let session_id = predictor.start_session(source, session_options)?;
    let num_frames = predictor.session_frame_count(&session_id)?;
    println!("Created video session {session_id} with {num_frames} frames");

    if video_mode.prompt_text.is_none()
        && video_mode.points.is_empty()
        && video_mode.boxes.is_empty()
    {
        bail!("video mode requires a prompt via --video-prompt, --point, or --box")
    }

    let obj_id = predictor.add_prompt(
        &session_id,
        0,
        sam3::SessionPrompt {
            text: video_mode.prompt_text.clone(),
            points: (!video_mode.points.is_empty()).then_some(video_mode.points.clone()),
            point_labels: (!video_mode.point_labels.is_empty())
                .then_some(video_mode.point_labels.clone()),
            boxes: (!video_mode.boxes.is_empty()).then_some(video_mode.boxes.clone()),
            box_labels: (!video_mode.box_labels.is_empty())
                .then_some(video_mode.box_labels.clone()),
        },
        None,
        true,
        true,
    )?;
    println!("Seeded object {obj_id} on frame 0");

    fs::create_dir_all(output_dir)?;
    let frames_dir = output_dir.join(VIDEO_FRAMES_DIR);
    let masks_dir = output_dir.join(VIDEO_MASKS_DIR);
    let masked_frames_dir = output_dir.join(VIDEO_MASKED_FRAMES_DIR);
    clear_output_dir(&frames_dir)?;
    clear_output_dir(&masks_dir)?;
    clear_output_dir(&masked_frames_dir)?;
    fs::create_dir_all(&frames_dir)?;
    fs::create_dir_all(&masks_dir)?;
    fs::create_dir_all(&masked_frames_dir)?;

    let mut export_source = ExportFrameSource::new(&source_path)?;
    let results_path = output_dir.join(VIDEO_RESULTS_FILE);
    let mut writer = std::io::BufWriter::new(fs::File::create(&results_path)?);
    writer.write_all(b"[\n")?;
    let mut wrote_any = false;
    let mut exported_frames = 0usize;

    predictor.propagate_in_video_stream(
        &session_id,
        sam3::PropagationOptions {
            direction: sam3::PropagationDirection::Forward,
            start_frame_idx: None,
            max_frame_num_to_track: None,
            output_prob_threshold: None,
        },
        |frame| {
            if frame.frame_idx % video_mode.frame_stride != 0 {
                return Ok(());
            }

            let frame_record = export_frame_record(
                frame,
                &mut export_source,
                output_dir,
                &frames_dir,
                &masks_dir,
                &masked_frames_dir,
            )
            .map_err(|err| candle::Error::Msg(err.to_string()))?;
            if wrote_any {
                writer.write_all(b",\n")?;
            }
            wrote_any = true;
            exported_frames += 1;
            serde_json::to_writer_pretty(&mut writer, &frame_record).map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to write {}: {}",
                    results_path.display(),
                    err
                ))
            })?;
            Ok(())
        },
    )?;

    writer.write_all(b"\n]\n")?;
    writer.flush()?;

    let metadata = VideoExportMetadata {
        bundle_version: default_bundle_version(),
        mode: "video_prediction_export".to_owned(),
        source_path: source_path.display().to_string(),
        source_kind: export_source.source_kind().to_owned(),
        session_frame_count: num_frames,
        exported_frame_count: exported_frames,
        frame_stride: video_mode.frame_stride.max(1),
        tokenizer_path: video_mode.tokenizer_path.clone(),
        prompt_text: video_mode.prompt_text.clone(),
        points_xy_normalized: video_mode
            .points
            .iter()
            .map(|(x, y)| vec![*x, *y])
            .collect(),
        point_labels: video_mode.point_labels.clone(),
        boxes_cxcywh_normalized: video_mode
            .boxes
            .iter()
            .map(|(cx, cy, w, h)| vec![*cx, *cy, *w, *h])
            .collect(),
        box_labels: video_mode.box_labels.clone(),
        frames_dir: VIDEO_FRAMES_DIR.to_owned(),
        masks_dir: VIDEO_MASKS_DIR.to_owned(),
        masked_frames_dir: VIDEO_MASKED_FRAMES_DIR.to_owned(),
        results_path: VIDEO_RESULTS_FILE.to_owned(),
        debug_dir: video_mode.debug_bundle.then(|| VIDEO_DEBUG_DIR.to_owned()),
    };
    let metadata_path = output_dir.join(VIDEO_REFERENCE_METADATA_FILE);
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    let stats = predictor.session_cache_stats(&session_id)?;
    println!(
        "Saved results to {} (loaded_frames={}, cached_features={}, cached_output_frames={}, tracked_objects={})",
        results_path.display(),
        stats.loaded_frame_count,
        stats.cached_feature_entries,
        stats.cached_output_frames,
        stats.tracked_objects
    );
    println!("Video export metadata: {}", metadata_path.display());

    predictor.close_session(&session_id)?;
    println!("Video prediction completed successfully.");
    Ok(())
}

pub fn is_video_reference_bundle(path: &Path) -> Result<bool> {
    let metadata_path = resolve_bundle_root(path).join(VIDEO_REFERENCE_METADATA_FILE);
    if !metadata_path.exists() {
        return Ok(false);
    }
    let metadata: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&metadata_path).with_context(|| {
            format!(
                "failed to read video reference metadata probe from {}",
                metadata_path.display()
            )
        })?)
        .with_context(|| {
            format!(
                "failed to parse video reference metadata probe from {}",
                metadata_path.display()
            )
        })?;
    Ok(metadata
        .get("mode")
        .and_then(|mode| mode.as_str())
        .map(|mode| mode.starts_with("video_"))
        .unwrap_or(false))
}

pub fn run_video_reference_comparison(
    model: &sam3::Sam3ImageModel,
    tracker: &sam3::Sam3TrackerModel,
    bundle_path: &str,
    output_dir: &Path,
    device: &Device,
    debug_bundle_override: bool,
    debug_obj_ids_override: &[u32],
    debug_frame_indices_override: &[usize],
) -> Result<()> {
    let (metadata, reference_results, bundle_root) = load_video_bundle(Path::new(bundle_path))?;
    let reference_frames_dir = bundle_root.join(&metadata.frames_dir);
    let reference_debug_root = metadata
        .debug_dir
        .as_ref()
        .map(|dir| bundle_root.join(dir))
        .filter(|path| path.join(VIDEO_DEBUG_MANIFEST_FILE).exists())
        .or_else(|| {
            let fallback = bundle_root.join(VIDEO_DEBUG_DIR);
            fallback
                .join(VIDEO_DEBUG_MANIFEST_FILE)
                .exists()
                .then_some(fallback)
        });
    let reference_debug = reference_debug_root
        .as_ref()
        .map(|path| load_video_debug_manifest(path))
        .transpose()?;
    let requested_debug_bundle = debug_bundle_override
        || !debug_obj_ids_override.is_empty()
        || !debug_frame_indices_override.is_empty();
    let debug_bundle = requested_debug_bundle || reference_debug.is_some();
    let debug_obj_ids = if !debug_obj_ids_override.is_empty() {
        debug_obj_ids_override.to_vec()
    } else {
        reference_debug
            .as_ref()
            .map(|manifest| manifest.capture_obj_ids.clone())
            .unwrap_or_default()
    };
    let debug_frame_indices = if !debug_frame_indices_override.is_empty() {
        debug_frame_indices_override.to_vec()
    } else {
        reference_debug
            .as_ref()
            .map(|manifest| manifest.capture_frame_indices.clone())
            .unwrap_or_default()
    };
    let video_mode = VideoMode {
        video_path: reference_frames_dir.display().to_string(),
        tokenizer_path: metadata.tokenizer_path.clone(),
        prompt_text: metadata.prompt_text.clone(),
        points: metadata
            .points_xy_normalized
            .iter()
            .filter_map(|point| (point.len() == 2).then_some((point[0], point[1])))
            .collect(),
        point_labels: metadata.point_labels.clone(),
        boxes: metadata
            .boxes_cxcywh_normalized
            .iter()
            .filter_map(|bbox| (bbox.len() == 4).then_some((bbox[0], bbox[1], bbox[2], bbox[3])))
            .collect(),
        box_labels: metadata.box_labels.clone(),
        frame_stride: metadata.frame_stride.max(1),
        prefetch_ahead: 2,
        prefetch_behind: 1,
        max_feature_cache_entries: 2,
        offload_state_to_cpu: false,
        debug_bundle,
        debug_obj_ids,
        debug_frame_indices,
    };

    run_video_prediction(model, tracker, &video_mode, output_dir, device)?;

    let (_actual_metadata, actual_results, actual_root) = load_video_bundle(output_dir)?;
    let report = build_video_reference_comparison_report(
        &bundle_root,
        bundle_path,
        &reference_results,
        &actual_root,
        output_dir,
        &actual_results,
    )?;
    let report_path = output_dir.join("video_reference_comparison_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!(
        "video reference comparison report written to {}",
        report_path.display()
    );
    if let (Some(reference_debug_root), Some(reference_debug)) =
        (reference_debug_root, reference_debug)
    {
        let actual_debug_root = output_dir.join(VIDEO_DEBUG_DIR);
        if actual_debug_root.join(VIDEO_DEBUG_MANIFEST_FILE).exists() {
            let actual_debug = load_video_debug_manifest(&actual_debug_root)?;
            let debug_report = build_video_debug_comparison_report(
                &reference_debug_root,
                bundle_path,
                &reference_debug,
                &actual_debug_root,
                output_dir,
                &actual_debug,
            )?;
            let debug_report_path = actual_debug_root.join(VIDEO_DEBUG_COMPARE_FILE);
            fs::create_dir_all(&actual_debug_root)?;
            fs::write(
                &debug_report_path,
                serde_json::to_string_pretty(&debug_report)?,
            )?;
            println!(
                "video debug comparison report written to {}",
                debug_report_path.display()
            );
        }
    }
    Ok(())
}

fn export_frame_record(
    frame: &sam3::VideoFrameOutput,
    frame_source: &mut ExportFrameSource,
    output_dir: &Path,
    frames_dir: &Path,
    masks_dir: &Path,
    masked_frames_dir: &Path,
) -> Result<VideoFrameRecord> {
    let frame_name = format!("frame_{:06}.png", frame.frame_idx);
    let frame_path = frames_dir.join(&frame_name);
    let base_frame = frame_source.load_rgba(frame.frame_idx)?;
    base_frame.save(&frame_path)?;

    let objects = frame
        .objects
        .iter()
        .map(|object| {
            export_object_record(
                frame.frame_idx,
                object,
                &base_frame,
                output_dir,
                masks_dir,
                masked_frames_dir,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(VideoFrameRecord {
        frame_idx: frame.frame_idx,
        frame_path: relative_output_path(output_dir, &frame_path),
        objects,
    })
}

fn export_object_record(
    frame_idx: usize,
    object: &sam3::ObjectFrameOutput,
    base_frame: &RgbaImage,
    output_dir: &Path,
    masks_dir: &Path,
    masked_frames_dir: &Path,
) -> Result<VideoObjectRecord> {
    let mask_probs = tensor_to_mask_probs(&object.masks)?;
    let mask_path = masks_dir.join(format!(
        "frame_{:06}_obj_{:06}.png",
        frame_idx, object.obj_id
    ));
    let masked_frame_path = masked_frames_dir.join(format!(
        "frame_{:06}_obj_{:06}.png",
        frame_idx, object.obj_id
    ));

    crate::threshold_mask(&mask_probs, MASK_THRESHOLD).save(&mask_path)?;

    let mut masked_frame = base_frame.clone();
    crate::blend_mask_with_threshold(&mut masked_frame, &mask_probs, MASK_COLOR, MASK_THRESHOLD);
    draw_segmentation_boxes(
        &mut masked_frame,
        &object.boxes_xyxy.to_vec2::<f32>()?,
        MASK_COLOR,
    );
    masked_frame.save(&masked_frame_path)?;

    Ok(VideoObjectRecord {
        obj_id: object.obj_id,
        scores: tensor_to_flat_vec(&object.scores)?,
        presence_scores: object
            .presence_scores
            .as_ref()
            .map(tensor_to_flat_vec)
            .transpose()?,
        boxes_xyxy: object.boxes_xyxy.to_vec2::<f32>()?,
        mask_path: Some(relative_output_path(output_dir, &mask_path)),
        masked_frame_path: Some(relative_output_path(output_dir, &masked_frame_path)),
        prompt_frame_idx: object.prompt_frame_idx,
        memory_frame_indices: object.memory_frame_indices.clone(),
        text_prompt: object.text_prompt.clone(),
        used_explicit_geometry: object.used_explicit_geometry,
        reused_previous_output: object.reused_previous_output,
    })
}

fn tensor_to_mask_probs(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let tensor = match tensor.rank() {
        2 => tensor.clone(),
        3 => tensor.i(0)?,
        4 => tensor.i((0, 0))?,
        rank => bail!("expected mask tensor rank 2/3/4, got {rank}"),
    };
    Ok(tensor.to_vec2::<f32>()?)
}

fn tensor_to_flat_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

fn relative_output_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn resolve_bundle_root(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

fn load_video_bundle(path: &Path) -> Result<(VideoExportMetadata, Vec<VideoFrameRecord>, PathBuf)> {
    let root = resolve_bundle_root(path);
    let metadata_path = root.join(VIDEO_REFERENCE_METADATA_FILE);
    let results_path = root.join(VIDEO_RESULTS_FILE);
    let metadata: VideoExportMetadata =
        serde_json::from_str(&fs::read_to_string(&metadata_path).with_context(|| {
            format!(
                "failed to read video metadata from {}",
                metadata_path.display()
            )
        })?)
        .with_context(|| {
            format!(
                "failed to parse video metadata from {}",
                metadata_path.display()
            )
        })?;
    let results: Vec<VideoFrameRecord> =
        serde_json::from_str(&fs::read_to_string(&results_path).with_context(|| {
            format!(
                "failed to read video results from {}",
                results_path.display()
            )
        })?)
        .with_context(|| {
            format!(
                "failed to parse video results from {}",
                results_path.display()
            )
        })?;
    Ok((metadata, results, root))
}

fn load_video_debug_manifest(path: &Path) -> Result<VideoDebugManifest> {
    let manifest_path = if path.is_dir() {
        path.join(VIDEO_DEBUG_MANIFEST_FILE)
    } else {
        path.to_path_buf()
    };
    serde_json::from_str(&fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "failed to read video debug manifest from {}",
            manifest_path.display()
        )
    })?)
    .with_context(|| {
        format!(
            "failed to parse video debug manifest from {}",
            manifest_path.display()
        )
    })
}

fn build_video_debug_comparison_report(
    reference_root: &Path,
    reference_bundle: &str,
    reference_manifest: &VideoDebugManifest,
    actual_root: &Path,
    actual_output_dir: &Path,
    actual_manifest: &VideoDebugManifest,
) -> Result<VideoDebugComparisonReport> {
    let candle_detector = actual_manifest
        .records
        .iter()
        .find(|record| record.stage == "detector_grounding");
    let candle_tracker_seed = actual_manifest
        .records
        .iter()
        .find(|record| record.stage == "tracker_seed");
    let candle_first_propagated = actual_manifest
        .records
        .iter()
        .find(|record| record.stage == "first_propagated_output");
    let reference_prompt = reference_manifest
        .records
        .iter()
        .find(|record| record.stage == "prompt_frame_output");
    let reference_first_propagated = reference_manifest
        .records
        .iter()
        .find(|record| record.stage == "first_propagated_output");

    let seed_frame_prompt_vs_detector = match (reference_prompt, candle_detector) {
        (Some(reference), Some(candle)) => Some(compare_debug_records(
            reference_root,
            Some(reference),
            actual_root,
            Some(candle),
        )?),
        _ => None,
    };
    let seed_frame_detector_vs_tracker_seed = match (candle_detector, candle_tracker_seed) {
        (Some(detector), Some(seed)) => Some(compare_debug_records(
            actual_root,
            Some(detector),
            actual_root,
            Some(seed),
        )?),
        _ => None,
    };
    let first_propagated_vs_reference = match (reference_first_propagated, candle_first_propagated)
    {
        (Some(reference), Some(candle)) => Some(compare_debug_records(
            reference_root,
            Some(reference),
            actual_root,
            Some(candle),
        )?),
        _ => None,
    };

    let candle_area_growth_ratio = area_growth_ratio(candle_detector, candle_first_propagated);
    let reference_area_growth_ratio =
        area_growth_ratio(reference_prompt, reference_first_propagated);
    let mut notes = Vec::new();
    let verdict = classify_debug_divergence(
        seed_frame_prompt_vs_detector.as_ref(),
        seed_frame_detector_vs_tracker_seed.as_ref(),
        first_propagated_vs_reference.as_ref(),
        candle_area_growth_ratio,
        reference_area_growth_ratio,
        &mut notes,
    );

    if !reference_manifest.internal_tracker_state_available {
        notes.push("upstream debug bundle is observable-output-only; internal tracker state is unavailable".to_owned());
    }

    Ok(VideoDebugComparisonReport {
        reference_bundle: reference_bundle.to_owned(),
        candle_output_dir: actual_output_dir.display().to_string(),
        seed_frame_prompt_vs_detector,
        seed_frame_detector_vs_tracker_seed,
        first_propagated_vs_reference,
        candle_area_growth_ratio,
        reference_area_growth_ratio,
        verdict,
        notes,
    })
}

fn compare_debug_records(
    lhs_root: &Path,
    lhs: Option<&VideoDebugRecord>,
    rhs_root: &Path,
    rhs: Option<&VideoDebugRecord>,
) -> Result<VideoDebugMetricSummary> {
    let lhs = lhs.context("missing lhs debug record for comparison")?;
    let rhs = rhs.context("missing rhs debug record for comparison")?;
    let mask_iou = match (
        lhs.observable
            .as_ref()
            .and_then(|summary| summary.mask_path.as_ref()),
        rhs.observable
            .as_ref()
            .and_then(|summary| summary.mask_path.as_ref()),
    ) {
        (Some(lhs_mask_path), Some(rhs_mask_path)) => {
            let lhs_mask = load_mask_probs(&lhs_root.join(lhs_mask_path))?;
            let rhs_mask = load_mask_probs(&rhs_root.join(rhs_mask_path))?;
            Some(crate::mask_iou_at_threshold(
                &lhs_mask,
                &rhs_mask,
                MASK_THRESHOLD,
                None,
            )?)
        }
        _ => None,
    };
    let lhs_box = lhs
        .observable
        .as_ref()
        .and_then(|summary| summary.boxes_xyxy.first())
        .cloned();
    let rhs_box = rhs
        .observable
        .as_ref()
        .and_then(|summary| summary.boxes_xyxy.first())
        .cloned();
    let (box_iou, mean_box_abs_diff) = match (lhs_box.as_deref(), rhs_box.as_deref()) {
        (Some(lhs_box), Some(rhs_box)) => {
            let mean_abs = lhs_box
                .iter()
                .zip(rhs_box.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .sum::<f32>()
                / lhs_box.len().max(1) as f32;
            (Some(crate::box_iou(lhs_box, rhs_box)), Some(mean_abs))
        }
        _ => (None, None),
    };
    Ok(VideoDebugMetricSummary {
        mask_iou_threshold_0_5: mask_iou,
        box_iou,
        mean_box_abs_diff,
    })
}

fn area_growth_ratio(
    first: Option<&VideoDebugRecord>,
    second: Option<&VideoDebugRecord>,
) -> Option<f32> {
    let first = first?.observable.as_ref()?;
    let second = second?.observable.as_ref()?;
    Some(second.mask_area_ratio / first.mask_area_ratio.max(1e-6))
}

fn classify_debug_divergence(
    seed_vs_reference: Option<&VideoDebugMetricSummary>,
    detector_vs_seed: Option<&VideoDebugMetricSummary>,
    propagated_vs_reference: Option<&VideoDebugMetricSummary>,
    candle_area_growth_ratio: Option<f32>,
    reference_area_growth_ratio: Option<f32>,
    notes: &mut Vec<String>,
) -> String {
    if let Some(metrics) = detector_vs_seed {
        if metrics.mask_iou_threshold_0_5.is_some_and(|iou| iou < 0.9)
            || metrics.box_iou.is_some_and(|iou| iou < 0.85)
            || metrics.mean_box_abs_diff.is_some_and(|diff| diff >= 0.02)
        {
            notes.push(
                "Candle tracker seed already diverges from Candle detector grounding on the prompt frame".to_owned(),
            );
            return "handoff".to_owned();
        }
    }
    if let Some(metrics) = seed_vs_reference {
        if metrics.mask_iou_threshold_0_5.is_some_and(|iou| iou < 0.9)
            || metrics.box_iou.is_some_and(|iou| iou < 0.85)
            || metrics.mean_box_abs_diff.is_some_and(|diff| diff >= 0.02)
        {
            notes.push("prompt-frame detector output is still not close to upstream".to_owned());
            return "handoff".to_owned();
        }
    }
    if let Some(metrics) = propagated_vs_reference {
        if let (Some(mask_iou), Some(box_iou), Some(box_abs_diff)) = (
            metrics.mask_iou_threshold_0_5,
            metrics.box_iou,
            metrics.mean_box_abs_diff,
        ) {
            if mask_iou >= 0.9 && (box_iou < 0.8 || box_abs_diff >= 0.05) {
                notes.push(
                    "propagated masks stay close while boxes diverge, pointing at postprocess/box extraction".to_owned(),
                );
                return "postprocess".to_owned();
            }
        }
    }
    if let Some(metrics) = propagated_vs_reference {
        if metrics.mask_iou_threshold_0_5.is_some_and(|iou| iou < 0.9) {
            notes.push("first propagated Candle mask diverges from upstream".to_owned());
            return "propagation".to_owned();
        }
    }
    if let (Some(candle_ratio), Some(reference_ratio)) =
        (candle_area_growth_ratio, reference_area_growth_ratio)
    {
        if candle_ratio > reference_ratio * 1.5 {
            notes.push(format!(
                "Candle mask area grows faster than upstream between the prompt frame and first propagated frame ({candle_ratio:.3} vs {reference_ratio:.3})"
            ));
            return "propagation".to_owned();
        }
    }
    notes.push(
        "no strong divergence signal was detected from the frame-0/frame-1 debug bundle".to_owned(),
    );
    "propagation".to_owned()
}

fn build_video_reference_comparison_report(
    reference_root: &Path,
    reference_bundle: &str,
    reference_frames: &[VideoFrameRecord],
    actual_root: &Path,
    actual_output_dir: &Path,
    actual_frames: &[VideoFrameRecord],
) -> Result<VideoReferenceComparisonReport> {
    let actual_by_frame = actual_frames
        .iter()
        .cloned()
        .map(|frame| (frame.frame_idx, frame))
        .collect::<BTreeMap<_, _>>();

    let mut all_reference_frames_present = true;
    let mut all_reference_objects_present = true;
    let mut score_diffs = Vec::new();
    let mut box_l1_diffs = Vec::new();
    let mut box_ious = Vec::new();
    let mut mask_abs_diffs = Vec::new();
    let mut mask_ious = Vec::new();
    let mut masked_frame_abs_diffs = Vec::new();
    let mut masked_frame_rmses = Vec::new();
    let mut frame_reports = Vec::with_capacity(reference_frames.len());

    for reference_frame in reference_frames {
        let Some(actual_frame) = actual_by_frame.get(&reference_frame.frame_idx) else {
            all_reference_frames_present = false;
            frame_reports.push(VideoFrameComparisonReport {
                frame_idx: reference_frame.frame_idx,
                reference_object_count: reference_frame.objects.len(),
                candle_object_count: 0,
                all_objects_present: false,
                object_reports: reference_frame
                    .objects
                    .iter()
                    .map(|object| VideoObjectComparisonReport {
                        obj_id: object.obj_id,
                        reference_score: first_score(object),
                        candle_score: None,
                        score_abs_diff: None,
                        reference_box_xyxy: first_box(object),
                        candle_box_xyxy: None,
                        box_l1_mean_abs_diff: None,
                        box_iou: None,
                        mask_mean_abs_diff: None,
                        mask_iou_threshold_0_5: None,
                        masked_frame_mean_abs_diff: None,
                        masked_frame_rmse: None,
                        notes: vec!["frame missing from Candle output".to_owned()],
                    })
                    .collect(),
            });
            continue;
        };

        let actual_by_obj = actual_frame
            .objects
            .iter()
            .cloned()
            .map(|object| (object.obj_id, object))
            .collect::<BTreeMap<_, _>>();
        let mut all_objects_present = true;
        let mut object_reports = Vec::with_capacity(reference_frame.objects.len());

        for reference_object in &reference_frame.objects {
            let mut notes = Vec::new();
            let actual_object = actual_by_obj.get(&reference_object.obj_id);
            if actual_object.is_none() {
                all_reference_objects_present = false;
                all_objects_present = false;
                notes.push("object missing from Candle output".to_owned());
            }

            let reference_score = first_score(reference_object);
            let candle_score = actual_object.and_then(first_score);
            let score_abs_diff = match (reference_score, candle_score) {
                (Some(reference), Some(actual)) => {
                    let diff = (reference - actual).abs();
                    score_diffs.push(diff);
                    Some(diff)
                }
                _ => None,
            };

            let reference_box = first_box(reference_object);
            let candle_box = actual_object.and_then(first_box);
            let (box_l1_mean_abs_diff, box_iou) =
                match (reference_box.as_ref(), candle_box.as_ref()) {
                    (Some(reference), Some(actual)) => {
                        let l1 = crate::mean_abs_box_diff(reference, actual);
                        let iou = crate::box_iou(reference, actual);
                        box_l1_diffs.push(l1);
                        box_ious.push(iou);
                        (Some(l1), Some(iou))
                    }
                    _ => (None, None),
                };

            let (mask_mean_abs_diff, mask_iou_threshold_0_5) =
                if let (Some(reference_mask_path), Some(actual_mask_path)) = (
                    reference_object.mask_path.as_ref(),
                    actual_object.and_then(|object| object.mask_path.as_ref()),
                ) {
                    let reference_mask =
                        load_mask_probs(&reference_root.join(reference_mask_path))?;
                    let actual_mask = load_mask_probs(&actual_root.join(actual_mask_path))?;
                    let abs_diff = crate::mask_mean_abs_diff(&reference_mask, &actual_mask, None)?;
                    let iou = crate::mask_iou_at_threshold(
                        &reference_mask,
                        &actual_mask,
                        MASK_THRESHOLD,
                        None,
                    )?;
                    mask_abs_diffs.push(abs_diff);
                    mask_ious.push(iou);
                    (Some(abs_diff), Some(iou))
                } else {
                    (None, None)
                };

            let (masked_frame_mean_abs_diff, masked_frame_rmse) =
                if let (Some(reference_masked_frame_path), Some(actual_masked_frame_path)) = (
                    reference_object.masked_frame_path.as_ref(),
                    actual_object.and_then(|object| object.masked_frame_path.as_ref()),
                ) {
                    let reference_image = crate::load_render_image(
                        &reference_root
                            .join(reference_masked_frame_path)
                            .display()
                            .to_string(),
                    )?;
                    let actual_image = crate::load_render_image(
                        &actual_root
                            .join(actual_masked_frame_path)
                            .display()
                            .to_string(),
                    )?;
                    let (mean_abs_diff, rmse) =
                        crate::image_diff_metrics(&reference_image, &actual_image, None)?;
                    masked_frame_abs_diffs.push(mean_abs_diff);
                    masked_frame_rmses.push(rmse);
                    (Some(mean_abs_diff), Some(rmse))
                } else {
                    (None, None)
                };

            object_reports.push(VideoObjectComparisonReport {
                obj_id: reference_object.obj_id,
                reference_score,
                candle_score,
                score_abs_diff,
                reference_box_xyxy: reference_box,
                candle_box_xyxy: candle_box,
                box_l1_mean_abs_diff,
                box_iou,
                mask_mean_abs_diff,
                mask_iou_threshold_0_5,
                masked_frame_mean_abs_diff,
                masked_frame_rmse,
                notes,
            });
        }

        frame_reports.push(VideoFrameComparisonReport {
            frame_idx: reference_frame.frame_idx,
            reference_object_count: reference_frame.objects.len(),
            candle_object_count: actual_frame.objects.len(),
            all_objects_present,
            object_reports,
        });
    }

    Ok(VideoReferenceComparisonReport {
        reference_bundle: reference_bundle.to_owned(),
        candle_output_dir: actual_output_dir.display().to_string(),
        reference_frame_count: reference_frames.len(),
        candle_frame_count: actual_frames.len(),
        compared_frame_count: frame_reports.len(),
        all_reference_frames_present,
        all_reference_objects_present,
        mean_score_abs_diff: mean_or_none(&score_diffs),
        mean_box_l1_abs_diff: mean_or_none(&box_l1_diffs),
        mean_box_iou: mean_or_none(&box_ious),
        mean_mask_abs_diff: mean_or_none(&mask_abs_diffs),
        mean_mask_iou_threshold_0_5: mean_or_none(&mask_ious),
        mean_masked_frame_abs_diff: mean_or_none(&masked_frame_abs_diffs),
        mean_masked_frame_rmse: mean_or_none(&masked_frame_rmses),
        frame_reports,
    })
}

fn first_score(object: &VideoObjectRecord) -> Option<f32> {
    object.scores.first().copied()
}

fn first_box(object: &VideoObjectRecord) -> Option<Vec<f32>> {
    object.boxes_xyxy.first().cloned()
}

fn mean_or_none(values: &[f32]) -> Option<f32> {
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

fn load_mask_probs(path: &Path) -> Result<Vec<Vec<f32>>> {
    let image = image::open(path)
        .with_context(|| format!("failed to open mask image {}", path.display()))?
        .to_luma8();
    let (width, height) = image.dimensions();
    let mut mask = vec![vec![0.0f32; width as usize]; height as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            mask[y][x] = f32::from(image.get_pixel(x as u32, y as u32)[0]) / 255.0;
        }
    }
    Ok(mask)
}

fn clear_output_dir(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)
            .with_context(|| format!("failed to clear output dir {}", path.display()))?;
    }
    Ok(())
}

fn draw_segmentation_boxes(image: &mut RgbaImage, boxes_xyxy: &[Vec<f32>], color: [u8; 3]) {
    let rgba = Rgba([color[0], color[1], color[2], 255]);
    let box_thickness = 3u32;
    for box_xyxy in boxes_xyxy {
        if box_xyxy.len() != 4 {
            continue;
        }
        let Some((x0, y0, x1, y1)) = normalized_box_to_pixel_bounds(
            [box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]],
            image.width(),
            image.height(),
        ) else {
            continue;
        };
        for offset in 0..box_thickness {
            let left = x0.saturating_sub(offset);
            let top = y0.saturating_sub(offset);
            let right = (x1 + offset).min(image.width().saturating_sub(1));
            let bottom = (y1 + offset).min(image.height().saturating_sub(1));
            draw_box_outline(image, left, top, right, bottom, rgba);
        }
    }
}

fn normalized_box_to_pixel_bounds(
    box_xyxy: [f32; 4],
    image_width: u32,
    image_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if image_width == 0 || image_height == 0 {
        return None;
    }
    let max_x = (image_width - 1) as f32;
    let max_y = (image_height - 1) as f32;
    let x0 = (box_xyxy[0].clamp(0.0, 1.0) * max_x).round() as u32;
    let y0 = (box_xyxy[1].clamp(0.0, 1.0) * max_y).round() as u32;
    let x1 = (box_xyxy[2].clamp(0.0, 1.0) * max_x).round() as u32;
    let y1 = (box_xyxy[3].clamp(0.0, 1.0) * max_y).round() as u32;
    if x1 < x0 || y1 < y0 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

fn draw_box_outline(
    image: &mut RgbaImage,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    color: Rgba<u8>,
) {
    for x in left..=right {
        image.put_pixel(x, top, color);
        image.put_pixel(x, bottom, color);
    }
    for y in top..=bottom {
        image.put_pixel(left, y, color);
        image.put_pixel(right, y, color);
    }
}

fn decode_video_frame_rgba(video_path: &Path, frame_idx: usize) -> Result<RgbaImage> {
    let select_filter = format!("select=eq(n\\,{frame_idx})");
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(video_path)
        .args([
            "-vf",
            &select_filter,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .output()
        .with_context(|| {
            format!(
                "failed to run ffmpeg for {} frame {}",
                video_path.display(),
                frame_idx
            )
        })?;
    if !output.status.success() {
        bail!(
            "ffmpeg failed for {} frame {}: {}",
            video_path.display(),
            frame_idx,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if output.stdout.is_empty() {
        bail!(
            "ffmpeg produced no bytes for {} frame {}",
            video_path.display(),
            frame_idx
        );
    }
    Ok(
        image::load(Cursor::new(output.stdout), image::ImageFormat::Png)
            .map_err(anyhow::Error::from)?
            .to_rgba8(),
    )
}

fn sorted_image_paths(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut image_paths = fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp"
                    )
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    image_paths.sort_by(|lhs, rhs| compare_image_paths(lhs, rhs));
    if image_paths.is_empty() {
        bail!("no image frames found in {}", dir_path.display())
    }
    Ok(image_paths)
}

fn compare_image_paths(lhs: &Path, rhs: &Path) -> std::cmp::Ordering {
    let lhs_stem = lhs
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    let rhs_stem = rhs
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    match (lhs_stem.parse::<usize>(), rhs_stem.parse::<usize>()) {
        (Ok(lhs_num), Ok(rhs_num)) => lhs_num.cmp(&rhs_num),
        _ => lhs_stem
            .cmp(rhs_stem)
            .then_with(|| lhs.file_name().cmp(&rhs.file_name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn temp_path(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("sam3-video-debug-{name}-{unique}"))
    }

    fn write_mask(path: &Path, rows: &[&[u8]]) -> Result<()> {
        let height = rows.len() as u32;
        let width = rows.first().map(|row| row.len()).unwrap_or(0) as u32;
        let mut image = GrayImage::new(width, height);
        for (y, row) in rows.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                image.put_pixel(x as u32, y as u32, Luma([*value]));
            }
        }
        image.save(path)?;
        Ok(())
    }

    fn debug_record(
        stage: &str,
        frame_idx: usize,
        obj_id: u32,
        mask_path: &str,
        box_xyxy: [f32; 4],
    ) -> VideoDebugRecord {
        VideoDebugRecord {
            stage: stage.to_owned(),
            obj_id,
            frame_idx,
            prompt_frame_idx: Some(0),
            prompt_metadata: None,
            observable: Some(VideoDebugObservableSummary {
                mask_path: Some(mask_path.to_owned()),
                mask_threshold: 0.5,
                foreground_pixel_count: 1,
                mask_area_ratio: 0.25,
                boxes_xyxy: vec![box_xyxy.into()],
                scores: vec![0.9],
                presence_scores: None,
                mask_logits_stats: TensorDebugSummary {
                    shape: vec![2, 2],
                    dtype: "u8".to_owned(),
                    min: 0.0,
                    max: 1.0,
                    mean: 0.25,
                    l2_norm: 1.0,
                    foreground_pixel_count: Some(1),
                },
                mask_prob_stats: TensorDebugSummary {
                    shape: vec![2, 2],
                    dtype: "u8".to_owned(),
                    min: 0.0,
                    max: 1.0,
                    mean: 0.25,
                    l2_norm: 1.0,
                    foreground_pixel_count: Some(1),
                },
            }),
            tracker_state: None,
            propagation_input: None,
        }
    }

    fn base_manifest(source: &str, records: Vec<VideoDebugRecord>) -> VideoDebugManifest {
        VideoDebugManifest {
            bundle_version: 1,
            mode: "video_debug_bundle".to_owned(),
            source: source.to_owned(),
            session_id: "session_0".to_owned(),
            internal_tracker_state_available: source == "candle",
            capture_obj_ids: Vec::new(),
            capture_frame_indices: vec![0, 1],
            capture_first_propagated_only: true,
            records,
        }
    }

    #[test]
    fn debug_comparison_detects_propagation_area_blow_up() -> Result<()> {
        let reference_root = temp_path("debug-report-reference");
        let actual_root = temp_path("debug-report-actual");
        fs::create_dir_all(&reference_root)?;
        fs::create_dir_all(&actual_root)?;
        write_mask(&reference_root.join("prompt.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&reference_root.join("prop.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("detector.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("seed.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("prop.png"), &[&[255, 255], &[255, 255]])?;

        let reference_manifest = base_manifest(
            "upstream",
            vec![
                debug_record(
                    "prompt_frame_output",
                    0,
                    0,
                    "prompt.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
                debug_record(
                    "first_propagated_output",
                    1,
                    0,
                    "prop.png",
                    [0.41, 0.5, 0.7, 0.8],
                ),
            ],
        );
        let actual_manifest = base_manifest(
            "candle",
            vec![
                debug_record(
                    "detector_grounding",
                    0,
                    0,
                    "detector.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
                debug_record("tracker_seed", 0, 0, "seed.png", [0.4, 0.5, 0.7, 0.8]),
                debug_record(
                    "first_propagated_output",
                    1,
                    0,
                    "prop.png",
                    [0.0, 0.0, 1.0, 1.0],
                ),
            ],
        );

        let report = build_video_debug_comparison_report(
            &reference_root,
            "reference_bundle",
            &reference_manifest,
            &actual_root,
            &actual_root,
            &actual_manifest,
        )?;
        assert_eq!(report.verdict, "propagation");
        Ok(())
    }

    #[test]
    fn debug_comparison_classifies_mask_close_box_far_as_postprocess() -> Result<()> {
        let reference_root = temp_path("debug-report-reference-postprocess");
        let actual_root = temp_path("debug-report-actual-postprocess");
        fs::create_dir_all(&reference_root)?;
        fs::create_dir_all(&actual_root)?;
        write_mask(&reference_root.join("prompt.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&reference_root.join("prop.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("detector.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("seed.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("prop.png"), &[&[255, 0], &[0, 0]])?;

        let reference_manifest = base_manifest(
            "upstream",
            vec![
                debug_record(
                    "prompt_frame_output",
                    0,
                    0,
                    "prompt.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
                debug_record(
                    "first_propagated_output",
                    1,
                    0,
                    "prop.png",
                    [0.41, 0.5, 0.7, 0.8],
                ),
            ],
        );
        let actual_manifest = base_manifest(
            "candle",
            vec![
                debug_record(
                    "detector_grounding",
                    0,
                    0,
                    "detector.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
                debug_record("tracker_seed", 0, 0, "seed.png", [0.4, 0.5, 0.7, 0.8]),
                debug_record(
                    "first_propagated_output",
                    1,
                    0,
                    "prop.png",
                    [0.0, 0.0, 1.0, 1.0],
                ),
            ],
        );

        let report = build_video_debug_comparison_report(
            &reference_root,
            "reference_bundle",
            &reference_manifest,
            &actual_root,
            &actual_root,
            &actual_manifest,
        )?;
        assert_eq!(report.verdict, "postprocess");
        Ok(())
    }

    #[test]
    fn debug_comparison_classifies_seed_box_drift_as_handoff() -> Result<()> {
        let reference_root = temp_path("debug-report-reference-handoff");
        let actual_root = temp_path("debug-report-actual-handoff");
        fs::create_dir_all(&reference_root)?;
        fs::create_dir_all(&actual_root)?;
        write_mask(&reference_root.join("prompt.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&reference_root.join("prop.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("detector.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("seed.png"), &[&[255, 0], &[0, 0]])?;
        write_mask(&actual_root.join("prop.png"), &[&[255, 0], &[0, 0]])?;

        let reference_manifest = base_manifest(
            "upstream",
            vec![
                debug_record(
                    "prompt_frame_output",
                    0,
                    0,
                    "prompt.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
                debug_record(
                    "first_propagated_output",
                    1,
                    0,
                    "prop.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
            ],
        );
        let actual_manifest = base_manifest(
            "candle",
            vec![
                debug_record(
                    "detector_grounding",
                    0,
                    0,
                    "detector.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
                debug_record("tracker_seed", 0, 0, "seed.png", [0.4, 0.5, 0.7, 1.0]),
                debug_record(
                    "first_propagated_output",
                    1,
                    0,
                    "prop.png",
                    [0.4, 0.5, 0.7, 0.8],
                ),
            ],
        );

        let report = build_video_debug_comparison_report(
            &reference_root,
            "reference_bundle",
            &reference_manifest,
            &actual_root,
            &actual_root,
            &actual_manifest,
        )?;
        assert_eq!(report.verdict, "handoff");
        Ok(())
    }
}
