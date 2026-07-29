// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle::Device;
use candle::{IndexOp, Tensor};
use candle_examples::sam3_video::{MediaFrameSource, PngVideoDebugArtifactSink};
use candle_transformers::models::sam3;
use clap::ValueEnum;
use image::{ImageReader, Rgb, RgbImage, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

const VIDEO_REFERENCE_METADATA_FILE: &str = "reference.json";
const VIDEO_RESULTS_FILE: &str = "video_results.json";
const VIDEO_FRAMES_DIR: &str = "frames";
const VIDEO_MASKS_DIR: &str = "masks";
const VIDEO_MASKED_FRAMES_DIR: &str = "masked_frames";
const VIDEO_COMBINED_FRAMES_DIR: &str = "combined_frames";
const VIDEO_CUTOUTS_RGBA_DIR: &str = "cutouts_rgba";
const VIDEO_CUTOUTS_RGB_BLACK_DIR: &str = "cutouts_rgb_black";
const VIDEO_DEBUG_DIR: &str = "debug";
const MASK_COLOR: [u8; 3] = [56, 201, 84];
const DEFAULT_MASK_THRESHOLD: f32 = 0.5;
const LEGACY_OVERLAY_ALPHA: f32 = 0.35;
const UPSTREAM_OVERLAY_ALPHA: f32 = 0.25;
const VIDEO_BUNDLE_VERSION: usize = 2;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum VideoRenderMode {
    /// Preserve the existing per-object green overlay artifact.
    PerObjectOverlay,
    /// Render every object on one frame with stable tab10 colors.
    CombinedUpstream,
    /// Write each object as RGBA with a transparent background.
    CutoutRgba,
    /// Write each object as RGB with a black background.
    CutoutRgbBlack,
}

pub struct VideoMode {
    pub video_path: String,
    pub tokenizer_path: Option<String>,
    pub prompt_text: Option<String>,
    pub points: Vec<(f32, f32)>,
    pub point_labels: Vec<u32>,
    pub boxes: Vec<(f32, f32, f32, f32)>,
    pub box_labels: Vec<u32>,
    pub frame_stride: usize,
    pub render_modes: Vec<VideoRenderMode>,
    pub mask_threshold: f32,
    pub draw_boxes: bool,
    pub draw_contours: bool,
    pub prefetch_ahead: usize,
    pub prefetch_behind: usize,
    pub max_feature_cache_entries: usize,
    pub offload_frames_to_cpu: bool,
    pub offload_state_to_cpu: bool,
    pub use_low_memory_profile: bool,
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
    #[serde(default = "default_render_modes")]
    render_modes: Vec<VideoRenderMode>,
    #[serde(default = "default_mask_threshold")]
    mask_threshold: f32,
    #[serde(default = "default_true")]
    draw_boxes: bool,
    #[serde(default)]
    draw_contours: bool,
    frames_dir: String,
    masks_dir: String,
    #[serde(default)]
    masked_frames_dir: Option<String>,
    #[serde(default)]
    combined_frames_dir: Option<String>,
    #[serde(default)]
    cutouts_rgba_dir: Option<String>,
    #[serde(default)]
    cutouts_rgb_black_dir: Option<String>,
    results_path: String,
    #[serde(default)]
    debug_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoFrameRecord {
    frame_idx: usize,
    frame_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    combined_frame_path: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cutout_rgba_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cutout_rgb_black_path: Option<String>,
    prompt_frame_idx: Option<usize>,
    memory_frame_indices: Vec<usize>,
    text_prompt: Option<String>,
    used_explicit_geometry: bool,
    reused_previous_output: bool,
}

struct VideoRenderOptions {
    modes: Vec<VideoRenderMode>,
    mask_threshold: f32,
    draw_boxes: bool,
    draw_contours: bool,
}

impl VideoRenderOptions {
    fn from_video_mode(video_mode: &VideoMode) -> Result<Self> {
        if !video_mode.mask_threshold.is_finite()
            || !(0.0..=1.0).contains(&video_mode.mask_threshold)
        {
            bail!("video mask threshold must be a finite value between 0 and 1")
        }
        let requested = if video_mode.render_modes.is_empty() {
            default_render_modes()
        } else {
            video_mode.render_modes.clone()
        };
        let mut modes = Vec::with_capacity(requested.len());
        for mode in requested {
            if !modes.contains(&mode) {
                modes.push(mode);
            }
        }
        Ok(Self {
            modes,
            mask_threshold: video_mode.mask_threshold,
            draw_boxes: video_mode.draw_boxes,
            draw_contours: video_mode.draw_contours,
        })
    }

    fn enabled(&self, mode: VideoRenderMode) -> bool {
        self.modes.contains(&mode)
    }
}

struct VideoExportDirs {
    frames: PathBuf,
    masks: PathBuf,
    masked_frames: Option<PathBuf>,
    combined_frames: Option<PathBuf>,
    cutouts_rgba: Option<PathBuf>,
    cutouts_rgb_black: Option<PathBuf>,
}

impl VideoExportDirs {
    fn prepare(output_dir: &Path, render: &VideoRenderOptions) -> Result<Self> {
        let frames = output_dir.join(VIDEO_FRAMES_DIR);
        let masks = output_dir.join(VIDEO_MASKS_DIR);
        let masked_frames = render
            .enabled(VideoRenderMode::PerObjectOverlay)
            .then(|| output_dir.join(VIDEO_MASKED_FRAMES_DIR));
        let combined_frames = render
            .enabled(VideoRenderMode::CombinedUpstream)
            .then(|| output_dir.join(VIDEO_COMBINED_FRAMES_DIR));
        let cutouts_rgba = render
            .enabled(VideoRenderMode::CutoutRgba)
            .then(|| output_dir.join(VIDEO_CUTOUTS_RGBA_DIR));
        let cutouts_rgb_black = render
            .enabled(VideoRenderMode::CutoutRgbBlack)
            .then(|| output_dir.join(VIDEO_CUTOUTS_RGB_BLACK_DIR));

        for dir_name in [
            VIDEO_FRAMES_DIR,
            VIDEO_MASKS_DIR,
            VIDEO_MASKED_FRAMES_DIR,
            VIDEO_COMBINED_FRAMES_DIR,
            VIDEO_CUTOUTS_RGBA_DIR,
            VIDEO_CUTOUTS_RGB_BLACK_DIR,
        ] {
            clear_output_dir(&output_dir.join(dir_name))?;
        }
        for dir in [
            Some(&frames),
            Some(&masks),
            masked_frames.as_ref(),
            combined_frames.as_ref(),
            cutouts_rgba.as_ref(),
            cutouts_rgb_black.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            fs::create_dir_all(dir)?;
        }

        Ok(Self {
            frames,
            masks,
            masked_frames,
            combined_frames,
            cutouts_rgba,
            cutouts_rgb_black,
        })
    }
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

fn default_render_modes() -> Vec<VideoRenderMode> {
    vec![VideoRenderMode::PerObjectOverlay]
}

fn default_mask_threshold() -> f32 {
    DEFAULT_MASK_THRESHOLD
}

fn default_true() -> bool {
    true
}

pub fn run_video_prediction(
    model: &sam3::Sam3ImageModel,
    tracker: &sam3::Sam3TrackerModel,
    video_mode: &VideoMode,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    println!("Starting video prediction for: {}", video_mode.video_path);

    let render = VideoRenderOptions::from_video_mode(video_mode)?;
    let frame_stride = video_mode.frame_stride.max(1);
    let source_path = PathBuf::from(&video_mode.video_path);
    let config = model.config();
    let source = MediaFrameSource::from_path(
        &video_mode.video_path,
        config.image.image_size,
        config.image.image_mean,
        config.image.image_std,
    )?;
    let memory_profile = if video_mode.use_low_memory_profile {
        sam3::VideoMemoryProfile::LowMemory
    } else {
        sam3::VideoMemoryProfile::Balanced
    };
    let session_options = sam3::VideoSessionOptions {
        tokenizer_path: video_mode.tokenizer_path.as_ref().map(PathBuf::from),
        memory_profile,
        offload_frames_to_cpu: video_mode.offload_frames_to_cpu,
        offload_state_to_cpu: video_mode.offload_state_to_cpu,
        retained_state_dtype: sam3::RetainedStateDType::F32,
        prefetch_ahead: video_mode.prefetch_ahead,
        prefetch_behind: video_mode.prefetch_behind,
        max_feature_cache_entries: video_mode.max_feature_cache_entries,
        max_non_cond_tracker_states: None,
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
            artifact_sink: video_mode.debug_bundle.then(|| {
                Arc::new(PngVideoDebugArtifactSink::new(debug_root.clone()))
                    as Arc<dyn sam3::VideoDebugArtifactSink>
            }),
        },
    );
    let session_id =
        predictor.start_session_with_frame_source(Box::new(source), session_options)?;
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
    let export_dirs = VideoExportDirs::prepare(output_dir, &render)?;

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
            if frame.frame_idx % frame_stride != 0 {
                return Ok(());
            }

            let frame_record =
                export_frame_record(frame, &mut export_source, output_dir, &export_dirs, &render)
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
        bundle_version: VIDEO_BUNDLE_VERSION,
        mode: "video_prediction_export".to_owned(),
        source_path: source_path.display().to_string(),
        source_kind: export_source.source_kind().to_owned(),
        session_frame_count: num_frames,
        exported_frame_count: exported_frames,
        frame_stride,
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
        render_modes: render.modes.clone(),
        mask_threshold: render.mask_threshold,
        draw_boxes: render.draw_boxes,
        draw_contours: render.draw_contours,
        frames_dir: VIDEO_FRAMES_DIR.to_owned(),
        masks_dir: VIDEO_MASKS_DIR.to_owned(),
        masked_frames_dir: export_dirs
            .masked_frames
            .as_ref()
            .map(|_| VIDEO_MASKED_FRAMES_DIR.to_owned()),
        combined_frames_dir: export_dirs
            .combined_frames
            .as_ref()
            .map(|_| VIDEO_COMBINED_FRAMES_DIR.to_owned()),
        cutouts_rgba_dir: export_dirs
            .cutouts_rgba
            .as_ref()
            .map(|_| VIDEO_CUTOUTS_RGBA_DIR.to_owned()),
        cutouts_rgb_black_dir: export_dirs
            .cutouts_rgb_black
            .as_ref()
            .map(|_| VIDEO_CUTOUTS_RGB_BLACK_DIR.to_owned()),
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
    export_dirs: &VideoExportDirs,
    render: &VideoRenderOptions,
) -> Result<VideoFrameRecord> {
    let frame_name = format!("frame_{:06}.png", frame.frame_idx);
    let frame_path = export_dirs.frames.join(&frame_name);
    let base_frame = frame_source.load_rgba(frame.frame_idx)?;
    base_frame.save(&frame_path)?;

    let mut objects = Vec::with_capacity(frame.objects.len());
    let mut combined_render_objects = export_dirs
        .combined_frames
        .as_ref()
        .map(|_| Vec::with_capacity(frame.objects.len()));
    for object in &frame.objects {
        let (record, mask_probs) = export_object_record(
            frame.frame_idx,
            object,
            &base_frame,
            output_dir,
            export_dirs,
            render,
        )?;
        if let Some(render_objects) = combined_render_objects.as_mut() {
            render_objects.push((object.obj_id, mask_probs, record.boxes_xyxy.clone()));
        }
        objects.push(record);
    }

    let combined_frame_path = export_dirs
        .combined_frames
        .as_ref()
        .map(|combined_frames_dir| -> Result<String> {
            let render_objects = combined_render_objects
                .as_ref()
                .expect("combined render objects should accompany combined output directory");
            let path = combined_frames_dir.join(&frame_name);
            let mut combined = base_frame.clone();
            for (obj_id, mask_probs, _) in render_objects {
                let color = object_color(*obj_id);
                blend_mask_with_alpha(
                    &mut combined,
                    mask_probs,
                    color,
                    render.mask_threshold,
                    UPSTREAM_OVERLAY_ALPHA,
                );
                if render.draw_contours {
                    draw_mask_contours(&mut combined, mask_probs, render.mask_threshold, color);
                }
            }
            if render.draw_boxes {
                for (obj_id, _, boxes_xyxy) in render_objects {
                    draw_segmentation_boxes(&mut combined, boxes_xyxy, object_color(*obj_id));
                }
            }
            combined.save(&path)?;
            Ok(relative_output_path(output_dir, &path))
        })
        .transpose()?;

    Ok(VideoFrameRecord {
        frame_idx: frame.frame_idx,
        frame_path: relative_output_path(output_dir, &frame_path),
        combined_frame_path,
        objects,
    })
}

fn export_object_record(
    frame_idx: usize,
    object: &sam3::ObjectFrameOutput,
    base_frame: &RgbaImage,
    output_dir: &Path,
    export_dirs: &VideoExportDirs,
    render: &VideoRenderOptions,
) -> Result<(VideoObjectRecord, Vec<Vec<f32>>)> {
    let mask_probs = tensor_to_mask_probs(&object.masks)?;
    validate_mask_dimensions(&mask_probs, base_frame)?;
    let artifact_name = format!("frame_{:06}_obj_{:06}.png", frame_idx, object.obj_id);
    let mask_path = export_dirs.masks.join(&artifact_name);

    crate::threshold_mask(&mask_probs, render.mask_threshold).save(&mask_path)?;

    let boxes_xyxy = object.boxes_xyxy.to_vec2::<f32>()?;
    let masked_frame_path = export_dirs
        .masked_frames
        .as_ref()
        .map(|masked_frames_dir| -> Result<String> {
            let path = masked_frames_dir.join(&artifact_name);
            let mut masked_frame = base_frame.clone();
            blend_mask_with_alpha(
                &mut masked_frame,
                &mask_probs,
                MASK_COLOR,
                render.mask_threshold,
                LEGACY_OVERLAY_ALPHA,
            );
            if render.draw_contours {
                draw_mask_contours(
                    &mut masked_frame,
                    &mask_probs,
                    render.mask_threshold,
                    MASK_COLOR,
                );
            }
            if render.draw_boxes {
                draw_segmentation_boxes(&mut masked_frame, &boxes_xyxy, MASK_COLOR);
            }
            masked_frame.save(&path)?;
            Ok(relative_output_path(output_dir, &path))
        })
        .transpose()?;

    let cutout_rgba_path = export_dirs
        .cutouts_rgba
        .as_ref()
        .map(|cutouts_dir| -> Result<String> {
            let path = cutouts_dir.join(&artifact_name);
            rgba_cutout(base_frame, &mask_probs, render.mask_threshold).save(&path)?;
            Ok(relative_output_path(output_dir, &path))
        })
        .transpose()?;

    let cutout_rgb_black_path = export_dirs
        .cutouts_rgb_black
        .as_ref()
        .map(|cutouts_dir| -> Result<String> {
            let path = cutouts_dir.join(&artifact_name);
            rgb_black_cutout(base_frame, &mask_probs, render.mask_threshold).save(&path)?;
            Ok(relative_output_path(output_dir, &path))
        })
        .transpose()?;

    let record = VideoObjectRecord {
        obj_id: object.obj_id,
        scores: tensor_to_flat_vec(&object.scores)?,
        presence_scores: object
            .presence_scores
            .as_ref()
            .map(tensor_to_flat_vec)
            .transpose()?,
        boxes_xyxy,
        mask_path: Some(relative_output_path(output_dir, &mask_path)),
        masked_frame_path,
        cutout_rgba_path,
        cutout_rgb_black_path,
        prompt_frame_idx: object.prompt_frame_idx,
        memory_frame_indices: object.memory_frame_indices.clone(),
        text_prompt: object.text_prompt.clone(),
        used_explicit_geometry: object.used_explicit_geometry,
        reused_previous_output: object.reused_previous_output,
    };
    Ok((record, mask_probs))
}

fn validate_mask_dimensions(mask_probs: &[Vec<f32>], frame: &RgbaImage) -> Result<()> {
    let expected_width = frame.width() as usize;
    let expected_height = frame.height() as usize;
    if mask_probs.len() != expected_height
        || mask_probs.iter().any(|row| row.len() != expected_width)
    {
        let actual_width = mask_probs.first().map(Vec::len).unwrap_or(0);
        bail!(
            "video mask dimensions {}x{} do not match source frame dimensions {}x{}",
            actual_width,
            mask_probs.len(),
            expected_width,
            expected_height
        )
    }
    Ok(())
}

fn object_color(obj_id: u32) -> [u8; 3] {
    const TAB10: [[u8; 3]; 10] = [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ];
    TAB10[obj_id as usize % TAB10.len()]
}

fn blend_mask_with_alpha(
    image: &mut RgbaImage,
    mask_probs: &[Vec<f32>],
    color: [u8; 3],
    threshold: f32,
    alpha: f32,
) {
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            if *prob >= threshold {
                let pixel = image.get_pixel_mut(x as u32, y as u32);
                pixel[0] = ((1.0 - alpha) * pixel[0] as f32 + alpha * color[0] as f32) as u8;
                pixel[1] = ((1.0 - alpha) * pixel[1] as f32 + alpha * color[1] as f32) as u8;
                pixel[2] = ((1.0 - alpha) * pixel[2] as f32 + alpha * color[2] as f32) as u8;
                pixel[3] = 255;
            }
        }
    }
}

fn rgba_cutout(frame: &RgbaImage, mask_probs: &[Vec<f32>], threshold: f32) -> RgbaImage {
    let mut cutout = RgbaImage::new(frame.width(), frame.height());
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            if *prob >= threshold {
                cutout.put_pixel(x as u32, y as u32, *frame.get_pixel(x as u32, y as u32));
            }
        }
    }
    cutout
}

fn rgb_black_cutout(frame: &RgbaImage, mask_probs: &[Vec<f32>], threshold: f32) -> RgbImage {
    let mut cutout = RgbImage::new(frame.width(), frame.height());
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            if *prob >= threshold {
                let pixel = frame.get_pixel(x as u32, y as u32);
                cutout.put_pixel(x as u32, y as u32, Rgb([pixel[0], pixel[1], pixel[2]]));
            }
        }
    }
    cutout
}

fn draw_mask_contours(
    image: &mut RgbaImage,
    mask_probs: &[Vec<f32>],
    threshold: f32,
    color: [u8; 3],
) {
    let height = mask_probs.len();
    let width = mask_probs.first().map(Vec::len).unwrap_or(0);
    let mut boundary = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if mask_probs[y][x] < threshold {
                continue;
            }
            let is_boundary = [
                (x as isize - 1, y as isize),
                (x as isize + 1, y as isize),
                (x as isize, y as isize - 1),
                (x as isize, y as isize + 1),
            ]
            .into_iter()
            .any(|(neighbor_x, neighbor_y)| {
                neighbor_x < 0
                    || neighbor_y < 0
                    || neighbor_x >= width as isize
                    || neighbor_y >= height as isize
                    || mask_probs[neighbor_y as usize][neighbor_x as usize] < threshold
            });
            if is_boundary {
                boundary.push((x as i32, y as i32));
            }
        }
    }

    draw_contour_stroke(image, &boundary, Rgba([255, 255, 255, 255]), 3);
    draw_contour_stroke(image, &boundary, Rgba([0, 0, 0, 255]), 2);
    draw_contour_stroke(
        image,
        &boundary,
        Rgba([color[0], color[1], color[2], 255]),
        1,
    );
}

fn draw_contour_stroke(
    image: &mut RgbaImage,
    boundary: &[(i32, i32)],
    color: Rgba<u8>,
    radius: i32,
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    for &(center_x, center_y) in boundary {
        for offset_y in -radius..=radius {
            for offset_x in -radius..=radius {
                if offset_x * offset_x + offset_y * offset_y > radius * radius {
                    continue;
                }
                let x = center_x + offset_x;
                let y = center_y + offset_y;
                if x >= 0 && y >= 0 && x < width && y < height {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn video_mode(render_modes: Vec<VideoRenderMode>) -> VideoMode {
        VideoMode {
            video_path: "frames".to_owned(),
            tokenizer_path: None,
            prompt_text: Some("person".to_owned()),
            points: Vec::new(),
            point_labels: Vec::new(),
            boxes: Vec::new(),
            box_labels: Vec::new(),
            frame_stride: 1,
            render_modes,
            mask_threshold: DEFAULT_MASK_THRESHOLD,
            draw_boxes: true,
            draw_contours: false,
            prefetch_ahead: 2,
            prefetch_behind: 1,
            max_feature_cache_entries: 2,
            offload_frames_to_cpu: false,
            offload_state_to_cpu: false,
            use_low_memory_profile: false,
            debug_bundle: false,
            debug_obj_ids: Vec::new(),
            debug_frame_indices: Vec::new(),
        }
    }

    fn temp_output_dir(test_name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("sam3_video_render_{test_name}_{unique}"))
    }

    #[test]
    fn render_options_preserve_legacy_default_and_deduplicate_modes() -> Result<()> {
        let default = VideoRenderOptions::from_video_mode(&video_mode(Vec::new()))?;
        assert_eq!(default.modes, vec![VideoRenderMode::PerObjectOverlay]);

        let selected = VideoRenderOptions::from_video_mode(&video_mode(vec![
            VideoRenderMode::CombinedUpstream,
            VideoRenderMode::CutoutRgba,
            VideoRenderMode::CombinedUpstream,
        ]))?;
        assert_eq!(
            selected.modes,
            vec![
                VideoRenderMode::CombinedUpstream,
                VideoRenderMode::CutoutRgba,
            ]
        );
        Ok(())
    }

    #[test]
    fn export_dirs_only_create_selected_artifact_directories() -> Result<()> {
        let output_dir = temp_output_dir("dirs");
        fs::create_dir_all(output_dir.join(VIDEO_MASKED_FRAMES_DIR))?;
        fs::write(
            output_dir.join(VIDEO_MASKED_FRAMES_DIR).join("stale.png"),
            [],
        )?;
        let render = VideoRenderOptions::from_video_mode(&video_mode(vec![
            VideoRenderMode::CombinedUpstream,
            VideoRenderMode::CutoutRgba,
        ]))?;

        let dirs = VideoExportDirs::prepare(&output_dir, &render)?;

        assert!(dirs.frames.is_dir());
        assert!(dirs.masks.is_dir());
        assert!(dirs
            .combined_frames
            .as_ref()
            .is_some_and(|dir| dir.is_dir()));
        assert!(dirs.cutouts_rgba.as_ref().is_some_and(|dir| dir.is_dir()));
        assert!(!output_dir.join(VIDEO_MASKED_FRAMES_DIR).exists());
        assert!(!output_dir.join(VIDEO_CUTOUTS_RGB_BLACK_DIR).exists());
        fs::remove_dir_all(output_dir)?;
        Ok(())
    }

    #[test]
    fn frame_export_writes_and_records_every_selected_artifact() -> Result<()> {
        let output_dir = temp_output_dir("frame_export");
        fs::create_dir_all(&output_dir)?;
        let source_path = output_dir.join("source.png");
        RgbaImage::from_pixel(4, 4, Rgba([20, 40, 60, 255])).save(&source_path)?;

        let mut mode = video_mode(vec![
            VideoRenderMode::PerObjectOverlay,
            VideoRenderMode::CombinedUpstream,
            VideoRenderMode::CutoutRgba,
            VideoRenderMode::CutoutRgbBlack,
        ]);
        mode.draw_boxes = false;
        let render = VideoRenderOptions::from_video_mode(&mode)?;
        let dirs = VideoExportDirs::prepare(&output_dir, &render)?;
        let mut source = ExportFrameSource::ImagePaths(vec![source_path]);
        let mask = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            (1, 1, 4, 4),
            &Device::Cpu,
        )?;
        let frame = sam3::VideoFrameOutput {
            frame_idx: 0,
            objects: vec![sam3::ObjectFrameOutput {
                obj_id: 3,
                mask_logits: mask.clone(),
                masks: mask,
                boxes_xyxy: Tensor::from_vec(
                    vec![0.25f32, 0.25, 0.75, 0.75],
                    (1, 4),
                    &Device::Cpu,
                )?,
                scores: Tensor::from_vec(vec![0.9f32], 1, &Device::Cpu)?,
                presence_scores: None,
                prompt_frame_idx: Some(0),
                memory_frame_indices: Vec::new(),
                text_prompt: Some("person".to_owned()),
                used_explicit_geometry: false,
                reused_previous_output: false,
            }],
        };

        let record = export_frame_record(&frame, &mut source, &output_dir, &dirs, &render)?;

        assert!(record
            .combined_frame_path
            .as_ref()
            .is_some_and(|path| output_dir.join(path).is_file()));
        let object = &record.objects[0];
        for path in [
            object.mask_path.as_ref(),
            object.masked_frame_path.as_ref(),
            object.cutout_rgba_path.as_ref(),
            object.cutout_rgb_black_path.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            assert!(output_dir.join(path).is_file(), "missing artifact {path}");
        }
        let serialized = serde_json::to_value(&record)?;
        assert!(serialized["combined_frame_path"].is_string());
        assert!(serialized["objects"][0]["cutout_rgba_path"].is_string());
        assert!(serialized["objects"][0]["cutout_rgb_black_path"].is_string());
        fs::remove_dir_all(output_dir)?;
        Ok(())
    }

    #[test]
    fn reference_metadata_serializes_render_configuration_and_directories() -> Result<()> {
        let metadata = VideoExportMetadata {
            bundle_version: VIDEO_BUNDLE_VERSION,
            mode: "video_prediction_export".to_owned(),
            source_path: "frames".to_owned(),
            source_kind: "image_folder".to_owned(),
            session_frame_count: 1,
            exported_frame_count: 1,
            frame_stride: 1,
            tokenizer_path: None,
            prompt_text: Some("person".to_owned()),
            points_xy_normalized: Vec::new(),
            point_labels: Vec::new(),
            boxes_cxcywh_normalized: Vec::new(),
            box_labels: Vec::new(),
            render_modes: vec![
                VideoRenderMode::CombinedUpstream,
                VideoRenderMode::CutoutRgba,
            ],
            mask_threshold: 0.6,
            draw_boxes: false,
            draw_contours: true,
            frames_dir: VIDEO_FRAMES_DIR.to_owned(),
            masks_dir: VIDEO_MASKS_DIR.to_owned(),
            masked_frames_dir: None,
            combined_frames_dir: Some(VIDEO_COMBINED_FRAMES_DIR.to_owned()),
            cutouts_rgba_dir: Some(VIDEO_CUTOUTS_RGBA_DIR.to_owned()),
            cutouts_rgb_black_dir: None,
            results_path: VIDEO_RESULTS_FILE.to_owned(),
            debug_dir: None,
        };

        let serialized = serde_json::to_value(metadata)?;

        assert_eq!(serialized["bundle_version"], VIDEO_BUNDLE_VERSION);
        assert_eq!(
            serialized["render_modes"],
            serde_json::json!(["combined-upstream", "cutout-rgba"])
        );
        assert_eq!(serialized["combined_frames_dir"], VIDEO_COMBINED_FRAMES_DIR);
        assert_eq!(serialized["cutouts_rgba_dir"], VIDEO_CUTOUTS_RGBA_DIR);
        let threshold = serialized["mask_threshold"]
            .as_f64()
            .expect("mask threshold should serialize as a number");
        assert!((threshold - 0.6).abs() < 1e-6);
        assert_eq!(serialized["draw_boxes"], false);
        assert_eq!(serialized["draw_contours"], true);
        Ok(())
    }

    #[test]
    fn cutout_renderers_apply_transparent_and_black_backgrounds() {
        let mut frame = RgbaImage::new(2, 2);
        frame.put_pixel(0, 0, Rgba([10, 20, 30, 255]));
        frame.put_pixel(1, 0, Rgba([40, 50, 60, 255]));
        frame.put_pixel(0, 1, Rgba([70, 80, 90, 255]));
        frame.put_pixel(1, 1, Rgba([100, 110, 120, 255]));
        let mask = vec![vec![0.9, 0.1], vec![0.5, 0.49]];

        let rgba = rgba_cutout(&frame, &mask, 0.5);
        assert_eq!(rgba.get_pixel(0, 0).0, [10, 20, 30, 255]);
        assert_eq!(rgba.get_pixel(1, 0).0, [0, 0, 0, 0]);
        assert_eq!(rgba.get_pixel(0, 1).0, [70, 80, 90, 255]);
        assert_eq!(rgba.get_pixel(1, 1).0, [0, 0, 0, 0]);

        let rgb = rgb_black_cutout(&frame, &mask, 0.5);
        assert_eq!(rgb.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(rgb.get_pixel(1, 0).0, [0, 0, 0]);
        assert_eq!(rgb.get_pixel(0, 1).0, [70, 80, 90]);
        assert_eq!(rgb.get_pixel(1, 1).0, [0, 0, 0]);
    }

    #[test]
    fn combined_overlay_uses_stable_tab10_color_and_upstream_alpha() {
        let mut frame = RgbaImage::from_pixel(1, 1, Rgba([100, 100, 100, 255]));
        blend_mask_with_alpha(
            &mut frame,
            &[vec![1.0]],
            object_color(0),
            0.5,
            UPSTREAM_OVERLAY_ALPHA,
        );
        assert_eq!(object_color(0), [31, 119, 180]);
        assert_eq!(object_color(10), object_color(0));
        assert_eq!(frame.get_pixel(0, 0).0, [82, 104, 120, 255]);
    }

    #[test]
    fn legacy_overlay_blend_remains_byte_compatible() {
        let mask = vec![vec![0.49, 0.5], vec![0.75, 1.0]];
        let original = RgbaImage::from_pixel(2, 2, Rgba([100, 120, 140, 255]));
        let mut expected = original.clone();
        crate::blend_mask_with_threshold(&mut expected, &mask, MASK_COLOR, DEFAULT_MASK_THRESHOLD);
        let mut actual = original;
        blend_mask_with_alpha(
            &mut actual,
            &mask,
            MASK_COLOR,
            DEFAULT_MASK_THRESHOLD,
            LEGACY_OVERLAY_ALPHA,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn contours_have_white_black_and_object_color_strokes() {
        let mut image = RgbaImage::from_pixel(15, 15, Rgba([10, 10, 10, 255]));
        let mut mask = vec![vec![0.0; 15]; 15];
        for row in mask.iter_mut().take(10).skip(6) {
            for prob in row.iter_mut().take(10).skip(6) {
                *prob = 1.0;
            }
        }
        let color = [31, 119, 180];

        draw_mask_contours(&mut image, &mask, 0.5, color);

        assert_eq!(image.get_pixel(7, 5).0, [31, 119, 180, 255]);
        assert_eq!(image.get_pixel(7, 4).0, [0, 0, 0, 255]);
        assert_eq!(image.get_pixel(7, 3).0, [255, 255, 255, 255]);
        assert_eq!(image.get_pixel(7, 2).0, [10, 10, 10, 255]);
    }

    #[test]
    fn mask_dimensions_must_match_export_frame() {
        let frame = RgbaImage::new(2, 2);
        let err = validate_mask_dimensions(&[vec![1.0, 1.0]], &frame)
            .expect_err("height mismatch should be rejected");
        assert!(err.to_string().contains("do not match"));
    }
}
