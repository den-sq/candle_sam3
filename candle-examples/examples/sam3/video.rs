// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

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
    pub offload_frames_to_cpu: bool,
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
        offload_frames_to_cpu: video_mode.offload_frames_to_cpu,
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
