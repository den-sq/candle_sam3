// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle::Device;
use candle_transformers::models::sam3;

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
}

pub fn run_video_prediction(
    model: &sam3::Sam3ImageModel,
    video_mode: &VideoMode,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    println!("Starting video prediction for: {}", video_mode.video_path);

    let source = sam3::VideoSource::from_path(&video_mode.video_path)?;
    let session_options = sam3::VideoSessionOptions {
        tokenizer_path: video_mode.tokenizer_path.as_ref().map(PathBuf::from),
        offload_frames_to_cpu: true,
        offload_state_to_cpu: video_mode.offload_state_to_cpu,
        prefetch_ahead: video_mode.prefetch_ahead,
        prefetch_behind: video_mode.prefetch_behind,
        max_feature_cache_entries: video_mode.max_feature_cache_entries,
    };

    let mut predictor = sam3::Sam3VideoPredictor::new(model, device);
    let session_id = predictor.start_session(source, session_options)?;
    let num_frames = predictor.session_frame_count(&session_id)?;
    println!("Created video session {session_id} with {num_frames} frames");

    if video_mode.prompt_text.is_none()
        && video_mode.points.is_empty()
        && video_mode.boxes.is_empty()
    {
        anyhow::bail!("video mode requires a prompt via --video-prompt, --point, or --box")
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

    std::fs::create_dir_all(output_dir)?;
    let results_path = output_dir.join("video_results.json");
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&results_path)?);
    writer.write_all(b"[\n")?;
    let mut wrote_any = false;

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
            if wrote_any {
                writer.write_all(b",\n")?;
            }
            wrote_any = true;
            let frame_json = serde_json::json!({
                "frame_idx": frame.frame_idx,
                "objects": frame.objects.iter().map(object_to_json).collect::<Vec<_>>(),
            });
            serde_json::to_writer_pretty(&mut writer, &frame_json).map_err(|err| {
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

    let stats = predictor.session_cache_stats(&session_id)?;
    println!(
        "Saved results to {} (loaded_frames={}, cached_features={}, cached_output_frames={}, tracked_objects={})",
        results_path.display(),
        stats.loaded_frame_count,
        stats.cached_feature_entries,
        stats.cached_output_frames,
        stats.tracked_objects
    );

    predictor.close_session(&session_id)?;
    println!("Video prediction completed successfully.");
    Ok(())
}

fn object_to_json(object: &sam3::ObjectFrameOutput) -> serde_json::Value {
    serde_json::json!({
        "obj_id": object.obj_id,
        "scores": object.scores.to_vec1::<f32>().unwrap_or_default(),
        "boxes_xyxy": object.boxes_xyxy.to_vec2::<f32>().unwrap_or_default(),
        "mask_shape": object.masks.dims(),
        "prompt_frame_idx": object.prompt_frame_idx,
        "memory_frame_indices": object.memory_frame_indices,
        "text_prompt": object.text_prompt,
        "used_explicit_geometry": object.used_explicit_geometry,
        "reused_previous_output": object.reused_previous_output,
    })
}
