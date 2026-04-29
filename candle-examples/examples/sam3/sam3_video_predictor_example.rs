use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle::{Device, IndexOp};
use candle_transformers::models::sam3;
use image::{ImageReader, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_rect_mut;
use serde_json::json;

const FRAME_STRIDE: usize = 60;
const MASK_THRESHOLD: f32 = 0.5;
const MASK_COLOR: [u8; 3] = [56, 201, 84];

pub(crate) fn run(
    model: &sam3::Sam3ImageModel,
    tracker: &sam3::Sam3TrackerModel,
    tokenizer_path: Option<&str>,
    notebook_asset_root: Option<&str>,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let asset_root = super::resolve_notebook_asset_root(notebook_asset_root)?;
    let video_path = asset_root.join("videos/0001");
    if !video_path.is_dir() {
        bail!(
            "video notebook expects a frame directory at {}",
            video_path.display()
        );
    }
    let tokenizer_path = tokenizer_path
        .map(ToOwned::to_owned)
        .context("video notebook example requires `--tokenizer <tokenizer.json>`")?;
    let frame_paths = sorted_frame_paths(&video_path)?;
    let first_frame = load_rgba_frame(&frame_paths, 0)?;
    let frame_width = first_frame.width() as f32;
    let frame_height = first_frame.height() as f32;

    let example_root = output_dir.join("sam3_video_predictor_example");
    clear_dir(&example_root)?;
    std::fs::create_dir_all(&example_root)?;
    std::fs::write(
        example_root.join("notebook_match.json"),
        serde_json::to_string_pretty(&json!({
            "notebook": "sam3_video_predictor_example.ipynb",
            "asset_root": asset_root.display().to_string(),
            "video_path": video_path.display().to_string(),
            "frame_count": frame_paths.len(),
            "frame_stride": FRAME_STRIDE,
            "runtime_note": "The Candle predictor currently tracks one object per add_prompt call, so the upstream remove_object(2) branch is executed only when object id 2 is present in the current runtime output.",
        }))?,
    )?;

    let source = sam3::VideoSource::from_path(
        video_path
            .to_str()
            .context("video notebook path is not valid UTF-8")?,
    )?;
    let session_options = sam3::VideoSessionOptions {
        tokenizer_path: Some(PathBuf::from(&tokenizer_path)),
        offload_frames_to_cpu: false,
        offload_state_to_cpu: false,
        prefetch_ahead: 2,
        prefetch_behind: 1,
        max_feature_cache_entries: 2,
    };
    let mut predictor = sam3::Sam3VideoPredictor::new(model, tracker, device);
    let session_id = predictor.start_session(source, session_options)?;
    predictor.reset_session(&session_id)?;

    let seed_obj_id = predictor.add_prompt(
        &session_id,
        0,
        sam3::SessionPrompt {
            text: Some("person".to_string()),
            points: None,
            point_labels: None,
            boxes: None,
            box_labels: None,
        },
        None,
        true,
        true,
    )?;

    let phase1_prompt_frame = propagate(
        &mut predictor,
        &session_id,
        Some(0),
        Some(0),
        Some(MASK_THRESHOLD),
    )?;
    export_phase(
        &example_root,
        "01_text_prompt_frame0",
        &frame_paths,
        &phase1_prompt_frame,
        1,
        json!({
            "phase": "text_prompt_frame0",
            "prompt": "person",
            "seed_obj_id": seed_obj_id,
        }),
    )?;

    let phase1_full = propagate(
        &mut predictor,
        &session_id,
        Some(0),
        None,
        Some(MASK_THRESHOLD),
    )?;
    let observed_obj_ids = export_phase(
        &example_root,
        "02_text_prompt_propagation",
        &frame_paths,
        &phase1_full,
        FRAME_STRIDE,
        json!({
            "phase": "text_prompt_propagation",
            "prompt": "person",
            "seed_obj_id": seed_obj_id,
        }),
    )?;

    let multi_object_target = 2u32;
    let remove_target_available = observed_obj_ids.contains(&multi_object_target);
    if remove_target_available {
        predictor.remove_object(&session_id, multi_object_target)?;
        let removed_full = propagate(
            &mut predictor,
            &session_id,
            Some(0),
            None,
            Some(MASK_THRESHOLD),
        )?;
        export_phase(
            &example_root,
            "03_remove_object_2_propagation",
            &frame_paths,
            &removed_full,
            FRAME_STRIDE,
            json!({
                "phase": "remove_object_2_propagation",
                "removed_obj_id": multi_object_target,
            }),
        )?;
    } else {
        write_phase_note(
            &example_root.join("03_remove_object_2_propagation"),
            json!({
                "phase": "remove_object_2_propagation",
                "status": "skipped",
                "reason": "Current Candle runtime did not materialize object id 2 for the upstream `person` text prompt, so the notebook remove_object(2) branch could not be reproduced exactly.",
                "observed_obj_ids": observed_obj_ids.iter().copied().collect::<Vec<_>>(),
            }),
        )?;
    }

    let tracked_obj_id = if remove_target_available {
        multi_object_target
    } else {
        seed_obj_id
    };
    let single_click_points = vec![(760.0 / frame_width, 550.0 / frame_height)];
    let single_click_labels = vec![1u32];
    predictor.add_prompt(
        &session_id,
        0,
        sam3::SessionPrompt {
            text: None,
            points: Some(single_click_points.clone()),
            point_labels: Some(single_click_labels.clone()),
            boxes: None,
            box_labels: None,
        },
        Some(tracked_obj_id),
        true,
        true,
    )?;
    let phase2_prompt_frame = propagate(
        &mut predictor,
        &session_id,
        Some(0),
        Some(0),
        Some(MASK_THRESHOLD),
    )?;
    export_phase(
        &example_root,
        "04_single_click_frame0",
        &frame_paths,
        &phase2_prompt_frame,
        1,
        json!({
            "phase": "single_click_frame0",
            "obj_id": tracked_obj_id,
            "points_xy_normalized": single_click_points,
            "point_labels": single_click_labels,
        }),
    )?;
    let phase2_full = propagate(
        &mut predictor,
        &session_id,
        Some(0),
        None,
        Some(MASK_THRESHOLD),
    )?;
    export_phase(
        &example_root,
        "05_single_click_propagation",
        &frame_paths,
        &phase2_full,
        FRAME_STRIDE,
        json!({
            "phase": "single_click_propagation",
            "obj_id": tracked_obj_id,
        }),
    )?;

    let refinement_points = vec![
        (740.0 / frame_width, 450.0 / frame_height),
        (760.0 / frame_width, 630.0 / frame_height),
        (840.0 / frame_width, 640.0 / frame_height),
        (760.0 / frame_width, 550.0 / frame_height),
    ];
    let refinement_labels = vec![1u32, 0, 0, 1];
    predictor.add_prompt(
        &session_id,
        0,
        sam3::SessionPrompt {
            text: None,
            points: Some(refinement_points.clone()),
            point_labels: Some(refinement_labels.clone()),
            boxes: None,
            box_labels: None,
        },
        Some(tracked_obj_id),
        true,
        true,
    )?;
    let phase3_prompt_frame = propagate(
        &mut predictor,
        &session_id,
        Some(0),
        Some(0),
        Some(MASK_THRESHOLD),
    )?;
    export_phase(
        &example_root,
        "06_refined_clicks_frame0",
        &frame_paths,
        &phase3_prompt_frame,
        1,
        json!({
            "phase": "refined_clicks_frame0",
            "obj_id": tracked_obj_id,
            "points_xy_normalized": refinement_points,
            "point_labels": refinement_labels,
        }),
    )?;
    let phase3_full = propagate(
        &mut predictor,
        &session_id,
        Some(0),
        None,
        Some(MASK_THRESHOLD),
    )?;
    export_phase(
        &example_root,
        "07_refined_clicks_propagation",
        &frame_paths,
        &phase3_full,
        FRAME_STRIDE,
        json!({
            "phase": "refined_clicks_propagation",
            "obj_id": tracked_obj_id,
        }),
    )?;

    predictor.close_session(&session_id)?;
    Ok(())
}

fn propagate(
    predictor: &mut sam3::Sam3VideoPredictor<'_>,
    session_id: &str,
    start_frame_idx: Option<usize>,
    max_frame_num_to_track: Option<usize>,
    output_prob_threshold: Option<f32>,
) -> Result<sam3::VideoOutput> {
    predictor
        .propagate_in_video(
            session_id,
            sam3::PropagationOptions {
                direction: sam3::PropagationDirection::Forward,
                start_frame_idx,
                max_frame_num_to_track,
                output_prob_threshold,
            },
        )
        .map_err(anyhow::Error::from)
}

fn export_phase(
    example_root: &Path,
    phase_name: &str,
    frame_paths: &[PathBuf],
    output: &sam3::VideoOutput,
    frame_stride: usize,
    metadata: serde_json::Value,
) -> Result<BTreeSet<u32>> {
    let phase_root = example_root.join(phase_name);
    clear_dir(&phase_root)?;
    let frames_dir = phase_root.join("frames");
    let masks_dir = phase_root.join("masks");
    let masked_frames_dir = phase_root.join("masked_frames");
    std::fs::create_dir_all(&frames_dir)?;
    std::fs::create_dir_all(&masks_dir)?;
    std::fs::create_dir_all(&masked_frames_dir)?;

    let mut exported_frames = Vec::new();
    let mut observed_obj_ids = BTreeSet::new();
    for frame in output
        .frames
        .iter()
        .filter(|frame| frame.frame_idx % frame_stride == 0)
    {
        let base_frame = load_rgba_frame(frame_paths, frame.frame_idx)?;
        let frame_path = frames_dir.join(format!("frame_{:06}.png", frame.frame_idx));
        base_frame.save(&frame_path)?;
        let mut object_records = Vec::new();
        for object in &frame.objects {
            observed_obj_ids.insert(object.obj_id);
            let mask_probs = tensor_to_mask_probs(&object.masks)?;
            let mask_path = masks_dir.join(format!(
                "frame_{:06}_obj_{:06}.png",
                frame.frame_idx, object.obj_id
            ));
            let masked_frame_path = masked_frames_dir.join(format!(
                "frame_{:06}_obj_{:06}.png",
                frame.frame_idx, object.obj_id
            ));
            super::threshold_mask(&mask_probs, MASK_THRESHOLD).save(&mask_path)?;
            let mut masked_frame = base_frame.clone();
            super::blend_mask_with_threshold(
                &mut masked_frame,
                &mask_probs,
                MASK_COLOR,
                MASK_THRESHOLD,
            );
            draw_boxes(&mut masked_frame, &object.boxes_xyxy.to_vec2::<f32>()?);
            masked_frame.save(&masked_frame_path)?;
            object_records.push(json!({
                "obj_id": object.obj_id,
                "scores": object.scores.flatten_all()?.to_vec1::<f32>()?,
                "presence_scores": object.presence_scores.as_ref().map(|scores| scores.flatten_all()?.to_vec1::<f32>()).transpose()?,
                "boxes_xyxy": object.boxes_xyxy.to_vec2::<f32>()?,
                "prompt_frame_idx": object.prompt_frame_idx,
                "memory_frame_indices": object.memory_frame_indices.clone(),
                "text_prompt": object.text_prompt.clone(),
                "used_explicit_geometry": object.used_explicit_geometry,
                "reused_previous_output": object.reused_previous_output,
                "mask_path": relative_output_path(&phase_root, &mask_path),
                "masked_frame_path": relative_output_path(&phase_root, &masked_frame_path),
            }));
        }
        exported_frames.push(json!({
            "frame_idx": frame.frame_idx,
            "frame_path": relative_output_path(&phase_root, &frame_path),
            "objects": object_records,
        }));
    }

    std::fs::write(
        phase_root.join("video_results.json"),
        serde_json::to_string_pretty(&exported_frames)?,
    )?;
    std::fs::write(
        phase_root.join("summary.json"),
        serde_json::to_string_pretty(&json!({
            "phase": phase_name,
            "frame_stride": frame_stride,
            "exported_frame_count": exported_frames.len(),
            "observed_obj_ids": observed_obj_ids.iter().copied().collect::<Vec<_>>(),
            "metadata": metadata,
        }))?,
    )?;
    Ok(observed_obj_ids)
}

fn write_phase_note(phase_root: &Path, note: serde_json::Value) -> Result<()> {
    clear_dir(phase_root)?;
    std::fs::create_dir_all(phase_root)?;
    std::fs::write(
        phase_root.join("summary.json"),
        serde_json::to_string_pretty(&note)?,
    )?;
    Ok(())
}

fn sorted_frame_paths(video_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = std::fs::read_dir(video_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("jpg"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort_by(|lhs, rhs| {
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
            _ => lhs_stem.cmp(rhs_stem),
        }
    });
    if paths.is_empty() {
        bail!("no .jpg frames found in {}", video_dir.display());
    }
    Ok(paths)
}

fn load_rgba_frame(frame_paths: &[PathBuf], frame_idx: usize) -> Result<RgbaImage> {
    let frame_path = frame_paths.get(frame_idx).ok_or_else(|| {
        anyhow::anyhow!(
            "frame index {} exceeds available frame count {}",
            frame_idx,
            frame_paths.len()
        )
    })?;
    Ok(ImageReader::open(frame_path)?
        .decode()
        .map_err(anyhow::Error::from)?
        .to_rgba8())
}

fn tensor_to_mask_probs(tensor: &candle::Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(match tensor.rank() {
        2 => tensor.to_vec2::<f32>()?,
        3 => tensor.i(0)?.to_vec2::<f32>()?,
        4 => tensor.i((0, 0))?.to_vec2::<f32>()?,
        rank => bail!("expected mask tensor rank 2/3/4, got {rank}"),
    })
}

fn draw_boxes(image: &mut RgbaImage, boxes_xyxy: &[Vec<f32>]) {
    let color = Rgba([MASK_COLOR[0], MASK_COLOR[1], MASK_COLOR[2], 255]);
    for box_xyxy in boxes_xyxy {
        if box_xyxy.len() != 4 {
            continue;
        }
        draw_hollow_rect_mut(
            image,
            super::normalized_box_to_rect(
                [box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]],
                image.width() as usize,
                image.height() as usize,
            ),
            color,
        );
    }
}

fn clear_dir(path: &Path) -> Result<()> {
    if path.exists() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("failed to clear {}", path.display()))?;
    }
    Ok(())
}

fn relative_output_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}
