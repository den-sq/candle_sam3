// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use anyhow::Result;
use candle::Device;
use candle_transformers::models::sam3;
use serde_json;
use std::path::Path;

pub struct VideoMode {
    pub video_path: String,
    pub prompt_text: Option<String>,
    pub points: Vec<(f32, f32)>,
    pub point_labels: Vec<u32>,
    pub boxes: Vec<(f32, f32, f32, f32)>,
    pub box_labels: Vec<u32>,
    pub frame_stride: usize,
}

impl VideoMode {
    pub fn new(video_path: String) -> Self {
        Self {
            video_path,
            prompt_text: None,
            points: Vec::new(),
            point_labels: Vec::new(),
            boxes: Vec::new(),
            box_labels: Vec::new(),
            frame_stride: 1,
        }
    }
}

/// Load video frames as tensors
/// Currently supports image sequences. For video files, install OpenCV dependencies:
/// - Ubuntu/Debian: apt-get install clang llvm-dev libopencv-dev
/// - macOS: brew install opencv
/// - Then uncomment opencv dependency in Cargo.toml
pub fn load_video_frames(
    video_path: &str,
    model: &sam3::Sam3ImageModel,
    device: &Device,
) -> Result<Vec<candle::Tensor>> {
    let path = Path::new(video_path);

    // Check if it's a directory (image sequence)
    if path.is_dir() {
        return load_image_sequence(video_path, model, device);
    }

    // Check file extension
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        match ext_str.as_str() {
            "mp4" | "avi" | "mov" | "mkv" | "webm" => {
                anyhow::bail!(
                    "Video file detected ({}). Video frame loading requires OpenCV.\n\
                     To enable video support:\n\
                     1. Install system dependencies:\n\
                        Ubuntu/Debian: sudo apt-get install clang llvm-dev libopencv-dev\n\
                        macOS: brew install opencv\n\
                        Other: Install clang, LLVM, and OpenCV development libraries\n\
                     2. Uncomment the opencv dependency in candle-examples/Cargo.toml\n\
                     3. Rebuild the project\n\
                     \n\
                     Alternatively, provide a directory containing image sequence (frame_0001.jpg, frame_0002.jpg, etc.)",
                    video_path
                );
            }
            "jpg" | "jpeg" | "png" | "bmp" | "tiff" => {
                // Single image - treat as single-frame video
                return load_single_image(video_path, model, device);
            }
            _ => {
                anyhow::bail!("Unsupported file format: {}. Supported: video files (with OpenCV) or image sequences", video_path);
            }
        }
    }

    anyhow::bail!("Invalid path: {}", video_path);
}

/// Load a sequence of images from a directory
fn load_image_sequence(
    dir_path: &str,
    model: &sam3::Sam3ImageModel,
    device: &Device,
) -> Result<Vec<candle::Tensor>> {
    use std::fs;

    println!("Loading image sequence from directory: {}", dir_path);

    let entries = fs::read_dir(dir_path)?;
    let mut image_files = Vec::new();

    // Collect image files and sort by name
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if matches!(ext_str.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "tiff") {
                image_files.push(path);
            }
        }
    }

    // Sort by filename for consistent ordering
    image_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if image_files.is_empty() {
        anyhow::bail!("No image files found in directory: {}", dir_path);
    }

    println!("Found {} image files", image_files.len());

    let mut frames = Vec::new();
    for (idx, image_path) in image_files.iter().enumerate() {
        match load_single_image_tensor(image_path.to_str().unwrap(), model, device) {
            Ok(tensor) => {
                frames.push(tensor);
                if idx % 10 == 0 {
                    println!("Loaded frame {}/{}", idx + 1, image_files.len());
                }
            }
            Err(e) => {
                println!("Warning: Failed to load {}: {}", image_path.display(), e);
            }
        }
    }

    if frames.is_empty() {
        anyhow::bail!("Failed to load any frames from directory: {}", dir_path);
    }

    println!(
        "Successfully loaded {} frames from image sequence",
        frames.len()
    );
    Ok(frames)
}

/// Load a single image as a tensor
fn load_single_image(
    image_path: &str,
    model: &sam3::Sam3ImageModel,
    device: &Device,
) -> Result<Vec<candle::Tensor>> {
    println!("Loading single image: {}", image_path);
    let tensor = load_single_image_tensor(image_path, model, device)?;
    Ok(vec![tensor])
}

/// Load a single image file and convert to tensor
fn load_single_image_tensor(
    image_path: &str,
    model: &sam3::Sam3ImageModel,
    device: &Device,
) -> Result<candle::Tensor> {
    crate::preprocess_image_path_exact(image_path, model, device).map_err(Into::into)
}

/// Run video prediction mode
pub fn run_video_prediction(
    model: &sam3::Sam3ImageModel,
    video_mode: &VideoMode,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    println!("Starting video prediction for: {}", &video_mode.video_path);

    // Load video frames
    let frames = load_video_frames(&video_mode.video_path, model, device)?;

    if frames.is_empty() {
        anyhow::bail!("No frames loaded from video: {}", video_mode.video_path);
    }

    // Create video predictor
    let mut predictor = sam3::Sam3VideoPredictor::new(model, device);

    // Start video session
    let session_id = predictor.start_session(frames)?;
    println!("Created video session: {}", session_id);

    let num_frames = predictor.session_frame_count(&session_id)?;
    println!("Created video session with {} frames", num_frames);

    // Add a prompt to frame 0 if any prompt data is present
    if video_mode.prompt_text.is_none()
        && video_mode.points.is_empty()
        && video_mode.boxes.is_empty()
    {
        anyhow::bail!("video mode requires a prompt via --video-prompt, --point, or --box")
    }

    if video_mode.prompt_text.is_some()
        && video_mode.points.is_empty()
        && video_mode.boxes.is_empty()
    {
        anyhow::bail!(
            "text-only video grounding is not yet implemented; provide points or boxes with --point/--box"
        );
    }

    let session_prompt = sam3::SessionPrompt {
        text: video_mode.prompt_text.clone(),
        points: if video_mode.points.is_empty() {
            None
        } else {
            Some(video_mode.points.clone())
        },
        point_labels: if video_mode.point_labels.is_empty() {
            None
        } else {
            Some(video_mode.point_labels.clone())
        },
        boxes: if video_mode.boxes.is_empty() {
            None
        } else {
            Some(video_mode.boxes.clone())
        },
        box_labels: if video_mode.box_labels.is_empty() {
            None
        } else {
            Some(video_mode.box_labels.clone())
        },
    };

    predictor.add_prompt(&session_id, 0, session_prompt)?;

    // Propagate through the video from frame 0 forward
    let outputs = predictor.propagate_in_video(&session_id, sam3::PropagationDirection::Forward)?;

    // Save results
    std::fs::create_dir_all(output_dir)?;
    let results_path = output_dir.join("video_results.json");

    let selected_outputs: Vec<_> = outputs
        .outputs_per_frame
        .iter()
        .filter(|(frame_idx, _)| *frame_idx % video_mode.frame_stride == 0)
        .collect();

    let mut results = Vec::new();
    for (frame_idx, grounding) in selected_outputs {
        if let (Ok(scores), Ok(boxes)) = (
            grounding.scores.to_vec1::<f32>(),
            grounding.boxes_xyxy.to_vec2::<f32>(),
        ) {
            results.push(serde_json::json!({
                "frame_idx": frame_idx,
                "scores": scores,
                "boxes": boxes,
                "mask_shape": grounding.masks.dims()
            }));
        } else {
            println!(
                "Warning: Failed to serialize results for frame {}",
                frame_idx
            );
        }
    }

    std::fs::write(&results_path, serde_json::to_string_pretty(&results)?)?;
    println!(
        "Saved {} frame results to: {}",
        results.len(),
        results_path.display()
    );

    predictor.close_session(&session_id)?;
    println!("Video prediction completed successfully!");
    Ok(())
}
