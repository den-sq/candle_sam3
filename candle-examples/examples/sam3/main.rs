#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod parity;

use anyhow::{bail, Context, Error as E, Result};
use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::sam3;
use image::{GrayImage, Luma, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

#[derive(Parser, Debug)]
struct Args {
    /// Optional path to the upstream `sam3.pt` checkpoint or a repo directory containing it.
    #[arg(long)]
    checkpoint: Option<String>,

    /// Optional path to a `tokenizer.json` or a repo directory containing it.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Optional image path for the vision and geometry smoke tests.
    #[arg(long)]
    image: Option<String>,

    /// Optional square resize used by the example smoke path before vision encoding.
    #[arg(long)]
    smoke_image_size: Option<usize>,

    /// Directory used for rendered overlay, mask, and summary outputs.
    #[arg(long, default_value = "candle-examples/examples/sam3/output")]
    output_dir: String,

    /// Optional parity bundle directory or `reference.safetensors` file.
    #[arg(long)]
    parity_bundle: Option<String>,

    /// Absolute tolerance used for stage-by-stage parity comparisons.
    #[arg(long, default_value_t = 1e-4f32)]
    parity_atol: f32,

    /// Optional JSON manifest for sequential notebook-style batch runs.
    #[arg(long)]
    batch_manifest: Option<String>,

    /// Run the canned scenarios from `examples/sam3_image_predictor_example.ipynb`.
    #[arg(long)]
    image_predictor_example: bool,

    /// Optional text prompt for the text-encoder smoke test.
    #[arg(long)]
    prompt: Option<String>,

    /// Repeated normalized point prompts in `x,y` format.
    #[arg(long = "point", value_parser = parse_point)]
    points: Vec<PointArg>,

    /// Optional repeated point labels aligned with `--point`, defaults to `1`.
    #[arg(long = "point-label")]
    point_labels: Vec<u32>,

    /// Repeated normalized box prompts in `cx,cy,w,h` format.
    #[arg(long = "box", value_parser = parse_box)]
    boxes: Vec<BoxArg>,

    /// Optional repeated box labels aligned with `--box`, defaults to `1`.
    #[arg(long = "box-label")]
    box_labels: Vec<u32>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    print_config: bool,
}

#[derive(Clone, Copy, Debug)]
struct PointArg {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
struct BoxArg {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

#[derive(Clone, Debug)]
struct GeometryInputs {
    points: Vec<PointArg>,
    point_labels: Vec<u32>,
    boxes: Vec<BoxArg>,
    box_labels: Vec<u32>,
    prompt: sam3::GeometryPrompt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RenderStyle {
    Combined,
    NotebookImagePredictor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreprocessMode {
    Exact,
    CropFill,
}

impl PreprocessMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::CropFill => "crop_fill",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResizeToFillTransform {
    resized_width: usize,
    resized_height: usize,
    crop_x: usize,
    crop_y: usize,
    target_size: usize,
    original_width: usize,
    original_height: usize,
}

impl ResizeToFillTransform {
    fn from_original(original_width: usize, original_height: usize, target_size: usize) -> Self {
        let scale = (target_size as f32 / original_width as f32)
            .max(target_size as f32 / original_height as f32);
        let resized_width = ((original_width as f32) * scale).round() as usize;
        let resized_height = ((original_height as f32) * scale).round() as usize;
        let crop_x = resized_width.saturating_sub(target_size) / 2;
        let crop_y = resized_height.saturating_sub(target_size) / 2;
        Self {
            resized_width,
            resized_height,
            crop_x,
            crop_y,
            target_size,
            original_width,
            original_height,
        }
    }

    fn map_box_to_original(self, box_xyxy: [f32; 4]) -> [f32; 4] {
        let target = self.target_size as f32;
        let resized_w = self.resized_width as f32;
        let resized_h = self.resized_height as f32;
        [
            ((self.crop_x as f32) + box_xyxy[0] * target) / resized_w,
            ((self.crop_y as f32) + box_xyxy[1] * target) / resized_h,
            ((self.crop_x as f32) + box_xyxy[2] * target) / resized_w,
            ((self.crop_y as f32) + box_xyxy[3] * target) / resized_h,
        ]
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BatchManifestFile {
    Jobs(Vec<BatchJob>),
    Named { jobs: Vec<BatchJob> },
}

#[derive(Debug, Deserialize)]
struct BatchJob {
    name: Option<String>,
    image: String,
    prompt: Option<String>,
    smoke_image_size: Option<usize>,
    #[serde(default)]
    points: Vec<BatchPoint>,
    #[serde(default)]
    boxes: Vec<BatchBox>,
}

#[derive(Debug, Deserialize)]
struct BatchPoint {
    x: f32,
    y: f32,
    #[serde(default = "default_positive_label")]
    label: u32,
}

#[derive(Debug, Deserialize)]
struct BatchBox {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    #[serde(default = "default_positive_label")]
    label: u32,
}

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

fn default_positive_label() -> u32 {
    1
}

fn parse_point(value: &str) -> std::result::Result<PointArg, String> {
    let coords = parse_floats(value, 2)?;
    Ok(PointArg {
        x: coords[0],
        y: coords[1],
    })
}

fn parse_box(value: &str) -> std::result::Result<BoxArg, String> {
    let coords = parse_floats(value, 4)?;
    Ok(BoxArg {
        cx: coords[0],
        cy: coords[1],
        w: coords[2],
        h: coords[3],
    })
}

fn parse_floats(value: &str, expected: usize) -> std::result::Result<Vec<f32>, String> {
    let parts = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| format!("failed to parse `{value}` as comma-separated floats: {err}"))?;
    if parts.len() != expected {
        return Err(format!(
            "expected {expected} comma-separated values, got {} in `{value}`",
            parts.len()
        ));
    }
    Ok(parts)
}

fn resolve_repo_file(path: &str, expected_file: &str) -> std::path::PathBuf {
    let path = PathBuf::from(path);
    if path.is_dir() {
        path.join(expected_file)
    } else {
        path
    }
}

fn get_tokenizer(tokenizer: &str, context_length: usize) -> Result<Tokenizer> {
    let tokenizer_path = resolve_repo_file(tokenizer, "tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        E::msg(format!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        ))
    })?;
    let pad_id = *tokenizer
        .get_vocab(true)
        .get(CLIP_EOT_TOKEN)
        .ok_or_else(|| {
            E::msg(format!(
                "tokenizer is missing required token `{CLIP_EOT_TOKEN}`"
            ))
        })?;
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: CLIP_EOT_TOKEN.to_string(),
        }))
        .with_truncation(Some(TruncationParams {
            max_length: context_length,
            ..Default::default()
        }))
        .map_err(E::msg)?;
    Ok(tokenizer)
}

fn tokenize_prompt(
    prompt: &str,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let encoding = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let input_ids = Tensor::new(vec![encoding.get_ids().to_vec()], device)?;
    let attention_mask = Tensor::new(vec![encoding.get_attention_mask().to_vec()], device)?;
    Ok((input_ids, attention_mask))
}

fn preprocess_image_for_sam3(
    image_path: &str,
    image_size: usize,
    config: &sam3::Config,
    preprocess_mode: PreprocessMode,
    device: &Device,
) -> Result<Tensor> {
    let data = match preprocess_mode {
        PreprocessMode::Exact => image::ImageReader::open(image_path)?
            .decode()
            .map_err(E::msg)?
            .resize_exact(
                image_size as u32,
                image_size as u32,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8()
            .into_raw(),
        PreprocessMode::CropFill => {
            candle_examples::load_image_and_resize(image_path, image_size, image_size)?
                .permute((1, 2, 0))?
                .flatten_all()?
                .to_vec1::<u8>()?
        }
    };
    let image =
        Tensor::from_vec(data, (image_size, image_size, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let image = image.to_device(device)?;
    let mean = Tensor::from_vec(config.image.image_mean.to_vec(), (3, 1, 1), device)?;
    let std = Tensor::from_vec(config.image.image_std.to_vec(), (3, 1, 1), device)?;
    let image = (image.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?
        .unsqueeze(0)?;
    Ok(image)
}

fn load_render_image(image_path: &str) -> Result<RgbaImage> {
    Ok(image::ImageReader::open(image_path)?
        .decode()
        .map_err(E::msg)?
        .to_rgba8())
}

fn best_kept_query(scores: &Tensor, threshold: f32) -> Result<(usize, f32)> {
    let scores = scores.to_vec3::<f32>()?;
    let mut best_kept: Option<(usize, f32)> = None;
    let mut best_any = (0usize, f32::NEG_INFINITY);
    for (idx, score) in scores[0].iter().enumerate() {
        let score = score[0];
        if score > best_any.1 {
            best_any = (idx, score);
        }
        if score > threshold {
            match best_kept {
                Some((_, best_score)) if best_score >= score => {}
                _ => best_kept = Some((idx, score)),
            }
        }
    }
    Ok(best_kept.unwrap_or(best_any))
}

fn normalized_box_to_rect(box_xyxy: [f32; 4], image_width: usize, image_height: usize) -> Rect {
    let x_scale = (image_width.saturating_sub(1)) as f32;
    let y_scale = (image_height.saturating_sub(1)) as f32;
    let x0 = (box_xyxy[0].clamp(0.0, 1.0) * x_scale).round() as i32;
    let y0 = (box_xyxy[1].clamp(0.0, 1.0) * y_scale).round() as i32;
    let x1 = (box_xyxy[2].clamp(0.0, 1.0) * x_scale).round() as i32;
    let y1 = (box_xyxy[3].clamp(0.0, 1.0) * y_scale).round() as i32;
    let min_x = x0.min(x1);
    let min_y = y0.min(y1);
    let width = (x1.max(x0) - min_x).max(1) as u32;
    let height = (y1.max(y0) - min_y).max(1) as u32;
    Rect::at(min_x, min_y).of_size(width, height)
}

fn cxcywh_to_xyxy(bbox: &BoxArg) -> [f32; 4] {
    [
        bbox.cx - bbox.w * 0.5,
        bbox.cy - bbox.h * 0.5,
        bbox.cx + bbox.w * 0.5,
        bbox.cy + bbox.h * 0.5,
    ]
}

fn blend_mask(image: &mut RgbaImage, mask_probs: &[Vec<f32>], color: [u8; 3]) -> Result<GrayImage> {
    let height = mask_probs.len();
    let width = mask_probs.first().map(|row| row.len()).unwrap_or(0);
    let mut mask = GrayImage::new(width as u32, height as u32);
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            let prob = prob.clamp(0.0, 1.0);
            let mask_value = (prob * 255.0).round() as u8;
            mask.put_pixel(x as u32, y as u32, Luma([mask_value]));
            if prob >= 0.5 {
                let pixel = image.get_pixel_mut(x as u32, y as u32);
                let alpha = 0.35f32;
                pixel[0] = ((1.0 - alpha) * pixel[0] as f32 + alpha * color[0] as f32) as u8;
                pixel[1] = ((1.0 - alpha) * pixel[1] as f32 + alpha * color[1] as f32) as u8;
                pixel[2] = ((1.0 - alpha) * pixel[2] as f32 + alpha * color[2] as f32) as u8;
                pixel[3] = 255;
            }
        }
    }
    Ok(mask)
}

fn restore_crop_fill_mask_probs(
    mask_probs: &[Vec<f32>],
    transform: ResizeToFillTransform,
) -> Result<Vec<Vec<f32>>> {
    let target_h = mask_probs.len();
    let target_w = mask_probs.first().map(|row| row.len()).unwrap_or(0);
    if target_h != transform.target_size || target_w != transform.target_size {
        bail!(
            "crop-fill mask restoration expected square mask size {}x{}, got {}x{}",
            transform.target_size,
            transform.target_size,
            target_w,
            target_h
        )
    }

    let mask_image = mask_probs_to_gray_image(mask_probs);
    let mut resized_canvas = GrayImage::new(
        transform.resized_width as u32,
        transform.resized_height as u32,
    );
    image::imageops::replace(
        &mut resized_canvas,
        &mask_image,
        transform.crop_x as i64,
        transform.crop_y as i64,
    );
    let restored = image::imageops::resize(
        &resized_canvas,
        transform.original_width as u32,
        transform.original_height as u32,
        image::imageops::FilterType::Triangle,
    );

    let mut out = vec![vec![0.0f32; transform.original_width]; transform.original_height];
    for (y, row) in out.iter_mut().enumerate() {
        for (x, value) in row.iter_mut().enumerate() {
            *value = f32::from(restored.get_pixel(x as u32, y as u32)[0]) / 255.0;
        }
    }
    Ok(out)
}

fn mask_probs_to_gray_image(mask_probs: &[Vec<f32>]) -> GrayImage {
    let height = mask_probs.len();
    let width = mask_probs.first().map(|row| row.len()).unwrap_or(0);
    let mut mask = GrayImage::new(width as u32, height as u32);
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            let mask_value = (prob.clamp(0.0, 1.0) * 255.0).round() as u8;
            mask.put_pixel(x as u32, y as u32, Luma([mask_value]));
        }
    }
    mask
}

fn threshold_mask(mask_probs: &[Vec<f32>], threshold: f32) -> GrayImage {
    let height = mask_probs.len();
    let width = mask_probs.first().map(|row| row.len()).unwrap_or(0);
    let mut mask = GrayImage::new(width as u32, height as u32);
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            let value = if *prob >= threshold { 255 } else { 0 };
            mask.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }
    mask
}

fn blend_mask_with_threshold(
    image: &mut RgbaImage,
    mask_probs: &[Vec<f32>],
    color: [u8; 3],
    threshold: f32,
) {
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            if *prob >= threshold {
                let pixel = image.get_pixel_mut(x as u32, y as u32);
                let alpha = 0.35f32;
                pixel[0] = ((1.0 - alpha) * pixel[0] as f32 + alpha * color[0] as f32) as u8;
                pixel[1] = ((1.0 - alpha) * pixel[1] as f32 + alpha * color[1] as f32) as u8;
                pixel[2] = ((1.0 - alpha) * pixel[2] as f32 + alpha * color[2] as f32) as u8;
                pixel[3] = 255;
            }
        }
    }
}

fn normalized_point_to_pixel(
    point: PointArg,
    image_width: usize,
    image_height: usize,
) -> (i32, i32) {
    let x_scale = (image_width.saturating_sub(1)) as f32;
    let y_scale = (image_height.saturating_sub(1)) as f32;
    let x = (point.x.clamp(0.0, 1.0) * x_scale).round() as i32;
    let y = (point.y.clamp(0.0, 1.0) * y_scale).round() as i32;
    (x, y)
}

fn decode_scores(decoder: &sam3::DecoderOutput) -> Result<Tensor> {
    let class_scores = decoder.pred_logits.apply(&candle_nn::ops::sigmoid)?;
    match &decoder.presence_logits {
        Some(presence_logits) => {
            let batch_size = presence_logits.dim(0)?;
            let presence_scores = presence_logits
                .apply(&candle_nn::ops::sigmoid)?
                .reshape((batch_size, 1, 1))?;
            Ok(class_scores.broadcast_mul(&presence_scores)?)
        }
        None => Ok(class_scores),
    }
}

fn prompt_color(label: u32, style: RenderStyle) -> Rgba<u8> {
    match style {
        RenderStyle::Combined => {
            if label == 0 {
                Rgba([239, 68, 68, 255])
            } else {
                Rgba([59, 130, 246, 255])
            }
        }
        RenderStyle::NotebookImagePredictor => {
            if label == 0 {
                Rgba([255, 0, 0, 255])
            } else {
                Rgba([0, 255, 0, 255])
            }
        }
    }
}

fn draw_prompt_annotations(
    image: &mut RgbaImage,
    input_points: &[PointArg],
    input_point_labels: &[u32],
    input_boxes: &[BoxArg],
    input_box_labels: &[u32],
    style: RenderStyle,
) {
    let image_width = image.width() as usize;
    let image_height = image.height() as usize;
    for (bbox, label) in input_boxes.iter().zip(input_box_labels.iter()) {
        draw_hollow_rect_mut(
            image,
            normalized_box_to_rect(cxcywh_to_xyxy(bbox), image_width, image_height),
            prompt_color(*label, style),
        );
    }
    for (point, label) in input_points.iter().zip(input_point_labels.iter()) {
        draw_filled_circle_mut(
            image,
            normalized_point_to_pixel(*point, image_width, image_height),
            5,
            prompt_color(*label, style),
        );
    }
}

fn save_render_outputs(
    image_path: &str,
    image_size: usize,
    output_dir: &Path,
    preprocess_mode: PreprocessMode,
    prompt_label: &str,
    text_prompt: Option<&str>,
    decoder: &sam3::DecoderOutput,
    segmentation: &sam3::SegmentationOutput,
    scores: &Tensor,
    input_points: &[PointArg],
    input_point_labels: &[u32],
    input_boxes: &[BoxArg],
    input_box_labels: &[u32],
    render_style: RenderStyle,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let mut overlay = load_render_image(image_path)?;
    let mut prediction_overlay = load_render_image(image_path)?;
    let render_width = overlay.width() as usize;
    let render_height = overlay.height() as usize;
    let crop_fill_transform = if preprocess_mode == PreprocessMode::CropFill {
        Some(ResizeToFillTransform::from_original(
            render_width,
            render_height,
            image_size,
        ))
    } else {
        None
    };
    draw_prompt_annotations(
        &mut overlay,
        input_points,
        input_point_labels,
        input_boxes,
        input_box_labels,
        render_style,
    );
    if matches!(render_style, RenderStyle::Combined) {
        draw_prompt_annotations(
            &mut prediction_overlay,
            input_points,
            input_point_labels,
            input_boxes,
            input_box_labels,
            render_style,
        );
    }

    let (best_idx, best_score) = best_kept_query(scores, DEFAULT_CONFIDENCE_THRESHOLD)?;
    let pred_boxes = decoder.pred_boxes.to_vec3::<f32>()?;
    let best_box_cxcywh = &pred_boxes[0][best_idx];
    let best_box_model = cxcywh_to_xyxy(&BoxArg {
        cx: best_box_cxcywh[0],
        cy: best_box_cxcywh[1],
        w: best_box_cxcywh[2],
        h: best_box_cxcywh[3],
    })
    .to_vec();
    let best_box = if let Some(transform) = crop_fill_transform {
        transform
            .map_box_to_original([
                best_box_model[0],
                best_box_model[1],
                best_box_model[2],
                best_box_model[3],
            ])
            .to_vec()
    } else {
        best_box_model.clone()
    };
    draw_hollow_rect_mut(
        &mut prediction_overlay,
        normalized_box_to_rect(
            [best_box[0], best_box[1], best_box[2], best_box[3]],
            render_width,
            render_height,
        ),
        Rgba([56, 201, 84, 255]),
    );

    let best_mask_logits = segmentation
        .mask_logits
        .i((0, best_idx))?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .upsample_bilinear2d(image_size, image_size, false)?
        .i((0, 0))?;
    let best_mask_probs = candle_nn::ops::sigmoid(&best_mask_logits)?;
    let best_mask_probs = best_mask_probs.to_vec2::<f32>()?;
    let best_mask_probs = if let Some(transform) = crop_fill_transform {
        restore_crop_fill_mask_probs(&best_mask_probs, transform)?
    } else {
        let best_mask_probs = Tensor::from_vec(
            best_mask_probs
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
            (image_size, image_size),
            &Device::Cpu,
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .upsample_bilinear2d(render_height, render_width, false)?
        .i((0, 0))?;
        best_mask_probs.to_vec2::<f32>()?
    };
    let inverted_mask_probs = best_mask_probs
        .iter()
        .map(|row| row.iter().map(|prob| 1.0f32 - prob).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mask = blend_mask(&mut prediction_overlay, &best_mask_probs, [56, 201, 84])?;

    if matches!(render_style, RenderStyle::Combined) {
        overlay = prediction_overlay.clone();
    }

    let overlay_path = output_dir.join("overlay.png");
    let prediction_overlay_path = output_dir.join("prediction_overlay.png");
    let mask_path = output_dir.join("mask.png");
    let mask_sigmoid_path = output_dir.join("mask_sigmoid.png");
    let mask_one_minus_sigmoid_path = output_dir.join("mask_one_minus_sigmoid.png");
    overlay.save(&overlay_path)?;
    prediction_overlay.save(&prediction_overlay_path)?;
    mask.save(&mask_path)?;
    mask_probs_to_gray_image(&best_mask_probs).save(&mask_sigmoid_path)?;
    mask_probs_to_gray_image(&inverted_mask_probs).save(&mask_one_minus_sigmoid_path)?;

    let thresholds = [0.5f32];
    let mut debug_masks = Vec::new();
    for threshold in thresholds {
        let suffix = format!("{:.1}", threshold).replace('.', "_");

        let sigmoid_threshold_mask_path =
            output_dir.join(format!("mask_sigmoid_threshold_{suffix}.png"));
        let one_minus_sigmoid_threshold_mask_path =
            output_dir.join(format!("mask_one_minus_sigmoid_threshold_{suffix}.png"));
        let sigmoid_overlay_path =
            output_dir.join(format!("prediction_overlay_sigmoid_threshold_{suffix}.png"));
        let one_minus_sigmoid_overlay_path = output_dir.join(format!(
            "prediction_overlay_one_minus_sigmoid_threshold_{suffix}.png"
        ));

        threshold_mask(&best_mask_probs, threshold).save(&sigmoid_threshold_mask_path)?;
        threshold_mask(&inverted_mask_probs, threshold)
            .save(&one_minus_sigmoid_threshold_mask_path)?;

        let mut sigmoid_overlay = load_render_image(image_path)?;
        draw_hollow_rect_mut(
            &mut sigmoid_overlay,
            normalized_box_to_rect(
                [best_box[0], best_box[1], best_box[2], best_box[3]],
                render_width,
                render_height,
            ),
            Rgba([56, 201, 84, 255]),
        );
        blend_mask_with_threshold(
            &mut sigmoid_overlay,
            &best_mask_probs,
            [56, 201, 84],
            threshold,
        );
        sigmoid_overlay.save(&sigmoid_overlay_path)?;

        let mut one_minus_sigmoid_overlay = load_render_image(image_path)?;
        draw_hollow_rect_mut(
            &mut one_minus_sigmoid_overlay,
            normalized_box_to_rect(
                [best_box[0], best_box[1], best_box[2], best_box[3]],
                render_width,
                render_height,
            ),
            Rgba([56, 201, 84, 255]),
        );
        blend_mask_with_threshold(
            &mut one_minus_sigmoid_overlay,
            &inverted_mask_probs,
            [56, 201, 84],
            threshold,
        );
        one_minus_sigmoid_overlay.save(&one_minus_sigmoid_overlay_path)?;

        debug_masks.push(json!({
            "threshold": threshold,
            "mask_sigmoid_threshold_path": sigmoid_threshold_mask_path.display().to_string(),
            "mask_one_minus_sigmoid_threshold_path": one_minus_sigmoid_threshold_mask_path.display().to_string(),
            "prediction_overlay_sigmoid_threshold_path": sigmoid_overlay_path.display().to_string(),
            "prediction_overlay_one_minus_sigmoid_threshold_path": one_minus_sigmoid_overlay_path.display().to_string(),
        }));
    }

    let summary = json!({
        "prompt_label": prompt_label,
        "text_prompt": text_prompt,
        "render_image_size": {
            "width": render_width,
            "height": render_height,
        },
        "preprocess_mode": preprocess_mode.as_str(),
        "model_input_size": image_size,
        "best_query_index": best_idx,
        "best_score": best_score,
        "best_box_xyxy_normalized": best_box,
        "input_points_xy_normalized": input_points.iter().map(|point| vec![point.x, point.y]).collect::<Vec<_>>(),
        "input_point_labels": input_point_labels,
        "input_boxes_cxcywh_normalized": input_boxes.iter().map(|bbox| vec![bbox.cx, bbox.cy, bbox.w, bbox.h]).collect::<Vec<_>>(),
        "input_box_labels": input_box_labels,
        "overlay_path": overlay_path.display().to_string(),
        "prediction_overlay_path": prediction_overlay_path.display().to_string(),
        "mask_path": mask_path.display().to_string(),
        "mask_sigmoid_path": mask_sigmoid_path.display().to_string(),
        "mask_one_minus_sigmoid_path": mask_one_minus_sigmoid_path.display().to_string(),
        "debug_masks": debug_masks,
    });
    let summary_path = output_dir.join("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!("rendered outputs:");
    println!("  preprocess mode: {}", preprocess_mode.as_str());
    println!("  best query index: {best_idx}");
    println!("  best score: {best_score:.4}");
    println!("  best box xyxy (normalized): {:?}", best_box);
    println!("  overlay: {}", overlay_path.display());
    println!(
        "  prediction overlay: {}",
        prediction_overlay_path.display()
    );
    println!("  mask: {}", mask_path.display());
    println!("  mask sigmoid: {}", mask_sigmoid_path.display());
    println!(
        "  mask one-minus-sigmoid: {}",
        mask_one_minus_sigmoid_path.display()
    );
    println!("  summary: {}", summary_path.display());
    Ok(())
}

fn build_geometry_prompt_from_parts(
    points: &[PointArg],
    point_labels: &[u32],
    boxes: &[BoxArg],
    box_labels: &[u32],
    device: &Device,
) -> Result<Option<GeometryInputs>> {
    if points.is_empty() && boxes.is_empty() {
        return Ok(None);
    }
    if !point_labels.is_empty() && point_labels.len() != points.len() {
        bail!(
            "`--point-label` count ({}) must match `--point` count ({})",
            point_labels.len(),
            points.len()
        )
    }
    if !box_labels.is_empty() && box_labels.len() != boxes.len() {
        bail!(
            "`--box-label` count ({}) must match `--box` count ({})",
            box_labels.len(),
            boxes.len()
        )
    }

    let resolved_point_labels = if points.is_empty() {
        Vec::new()
    } else if point_labels.is_empty() {
        vec![1u32; points.len()]
    } else {
        point_labels.to_vec()
    };

    let resolved_box_labels = if boxes.is_empty() {
        Vec::new()
    } else if box_labels.is_empty() {
        vec![1u32; boxes.len()]
    } else {
        box_labels.to_vec()
    };

    let points_xy = if points.is_empty() {
        None
    } else {
        let data = points
            .iter()
            .flat_map(|point| [point.x, point.y])
            .collect::<Vec<_>>();
        Some(Tensor::from_vec(data, (points.len(), 2), device)?)
    };
    let point_labels = if points.is_empty() {
        None
    } else {
        Some(Tensor::new(resolved_point_labels.clone(), device)?)
    };

    let boxes_cxcywh = if boxes.is_empty() {
        None
    } else {
        let data = boxes
            .iter()
            .flat_map(|bbox| [bbox.cx, bbox.cy, bbox.w, bbox.h])
            .collect::<Vec<_>>();
        Some(Tensor::from_vec(data, (boxes.len(), 4), device)?)
    };
    let box_labels = if boxes.is_empty() {
        None
    } else {
        Some(Tensor::new(resolved_box_labels.clone(), device)?)
    };

    Ok(Some(GeometryInputs {
        points: points.to_vec(),
        point_labels: resolved_point_labels,
        boxes: boxes.to_vec(),
        box_labels: resolved_box_labels,
        prompt: sam3::GeometryPrompt {
            boxes_cxcywh,
            box_labels,
            points_xy,
            point_labels,
            masks: None,
            mask_labels: None,
        },
    }))
}

fn geometry_inputs_from_cli(args: &Args, device: &Device) -> Result<Option<GeometryInputs>> {
    build_geometry_prompt_from_parts(
        &args.points,
        &args.point_labels,
        &args.boxes,
        &args.box_labels,
        device,
    )
}

fn prompt_label(text_prompt: Option<&str>, geometry_inputs: Option<&GeometryInputs>) -> String {
    match (text_prompt, geometry_inputs.is_some()) {
        (Some(prompt), true) => format!("{prompt} + geometry prompts"),
        (Some(prompt), false) => prompt.to_string(),
        (None, true) => "geometry prompts".to_string(),
        (None, false) => "no prompt".to_string(),
    }
}

fn combine_encoded_prompts(
    text_encoding: Option<&sam3::TextEncoding>,
    geometry_encoding: Option<&sam3::EncodedPrompt>,
) -> Result<Option<sam3::EncodedPrompt>> {
    match (text_encoding, geometry_encoding) {
        (Some(text), Some(geometry)) => Ok(Some(sam3::EncodedPrompt {
            features: Tensor::cat(&[&text.memory, &geometry.features], 0)?,
            padding_mask: Tensor::cat(&[&text.attention_mask, &geometry.padding_mask], 1)?,
        })),
        (Some(text), None) => Ok(Some(sam3::EncodedPrompt {
            features: text.memory.clone(),
            padding_mask: text.attention_mask.clone(),
        })),
        (None, Some(geometry)) => Ok(Some(sam3::EncodedPrompt {
            features: geometry.features.clone(),
            padding_mask: geometry.padding_mask.clone(),
        })),
        (None, None) => Ok(None),
    }
}

fn load_batch_manifest(path: &str) -> Result<Vec<BatchJob>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read batch manifest from {path}"))?;
    let manifest = serde_json::from_str::<BatchManifestFile>(&raw)
        .with_context(|| format!("failed to parse batch manifest JSON from {path}"))?;
    let jobs = match manifest {
        BatchManifestFile::Jobs(jobs) => jobs,
        BatchManifestFile::Named { jobs } => jobs,
    };
    if jobs.is_empty() {
        bail!("batch manifest `{path}` does not contain any jobs")
    }
    Ok(jobs)
}

fn image_predictor_example_jobs() -> Vec<BatchJob> {
    vec![
        BatchJob {
            name: Some("image_predictor_text_shoe".to_string()),
            image: "/home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg"
                .to_string(),
            prompt: Some("shoe".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![],
        },
        BatchJob {
            name: Some("image_predictor_single_positive_box".to_string()),
            image: "/home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg"
                .to_string(),
            prompt: None,
            smoke_image_size: None,
            points: vec![],
            boxes: vec![BatchBox {
                cx: 0.41796875,
                cy: 0.6527777777777778,
                w: 0.0859375,
                h: 0.5,
                label: 1,
            }],
        },
        BatchJob {
            name: Some("image_predictor_positive_negative_boxes".to_string()),
            image: "/home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg"
                .to_string(),
            prompt: None,
            smoke_image_size: None,
            points: vec![],
            boxes: vec![
                BatchBox {
                    cx: 0.41796875,
                    cy: 0.6527777777777778,
                    w: 0.0859375,
                    h: 0.5,
                    label: 1,
                },
                BatchBox {
                    cx: 0.333984375,
                    cy: 0.6493055555555556,
                    w: 0.08984375,
                    h: 0.5208333333333334,
                    label: 0,
                },
            ],
        },
    ]
}

fn geometry_inputs_from_job(job: &BatchJob, device: &Device) -> Result<Option<GeometryInputs>> {
    let points = job
        .points
        .iter()
        .map(|point| PointArg {
            x: point.x,
            y: point.y,
        })
        .collect::<Vec<_>>();
    let point_labels = job
        .points
        .iter()
        .map(|point| point.label)
        .collect::<Vec<_>>();
    let boxes = job
        .boxes
        .iter()
        .map(|bbox| BoxArg {
            cx: bbox.cx,
            cy: bbox.cy,
            w: bbox.w,
            h: bbox.h,
        })
        .collect::<Vec<_>>();
    let box_labels = job.boxes.iter().map(|bbox| bbox.label).collect::<Vec<_>>();
    build_geometry_prompt_from_parts(&points, &point_labels, &boxes, &box_labels, device)
}

fn sanitize_job_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('-').to_string()
}

fn run_text_encoder(
    model: &sam3::Sam3ImageModel,
    prompt: &str,
    tokenizer_path: &str,
    context_length: usize,
    device: &Device,
) -> Result<sam3::TextEncoding> {
    let tokenizer = get_tokenizer(tokenizer_path, context_length)?;
    let (input_ids, attention_mask) = tokenize_prompt(prompt, &tokenizer, device)?;
    let encoding = model.encode_text_tokens(&input_ids, &attention_mask)?;
    println!("text stage:");
    println!("  text: {prompt}");
    println!("  input_ids: {:?}", input_ids.to_vec2::<u32>()?);
    println!("  attention_mask: {:?}", attention_mask.to_vec2::<u32>()?);
    println!("  padding mask shape: {:?}", encoding.attention_mask.dims());
    println!(
        "  input embeddings shape: {:?}",
        encoding.input_embeddings.dims()
    );
    println!("  resized memory shape: {:?}", encoding.memory.dims());
    Ok(encoding)
}

fn run_vision_and_geometry(
    model: &sam3::Sam3ImageModel,
    image_path: &str,
    smoke_image_size: Option<usize>,
    output_dir: &Path,
    preprocess_mode: PreprocessMode,
    text_prompt: Option<&str>,
    text_encoding: Option<&sam3::TextEncoding>,
    geometry_inputs: Option<&GeometryInputs>,
    render_style: RenderStyle,
    device: &Device,
) -> Result<()> {
    let config = model.config();
    let (original_image, initial_h, initial_w) = candle_examples::load_image(image_path, None)?;
    let mut state = model.set_image(&original_image)?;
    if let Some(text_prompt) = text_prompt {
        state = state.with_text_prompt(text_prompt.to_string());
    }
    if let Some(geometry_inputs) = geometry_inputs {
        state = state.with_geometry_prompt(geometry_inputs.prompt.clone());
    }
    println!("typed image state:");
    println!("  original image size: {}x{}", initial_h, initial_w);
    println!(
        "  model input size: {}x{}",
        state.model_input_size.height, state.model_input_size.width
    );
    println!("  has text prompt: {}", state.text_prompt().is_some());
    println!(
        "  has geometry prompt: {}",
        !state.geometry_prompt().is_empty()
    );

    let image_size = smoke_image_size.unwrap_or(config.image.image_size);
    let image = preprocess_image_for_sam3(image_path, image_size, config, preprocess_mode, device)?;
    println!("vision stage:");
    println!("  preprocessed image shape: {:?}", image.dims());
    println!("  smoke resize: {image_size}x{image_size}");
    println!("  preprocess mode: {}", preprocess_mode.as_str());
    let visual = model.encode_image_features(&image)?;
    println!("  backbone_fpn levels: {}", visual.backbone_fpn.len());
    for (level_idx, (features, pos)) in visual
        .backbone_fpn
        .iter()
        .zip(visual.vision_pos_enc.iter())
        .enumerate()
    {
        println!(
            "  level {level_idx}: features {:?}, pos {:?}",
            features.dims(),
            pos.dims()
        );
    }
    println!(
        "  sam2 side neck present: {}",
        visual.sam2_backbone_fpn.is_some()
    );

    let empty_geometry = sam3::GeometryPrompt::default();
    let empty_encoded = model.encode_geometry_prompt(&empty_geometry, &visual)?;
    println!("geometry stage:");
    println!(
        "  empty prompt: features {:?}, padding mask {:?}",
        empty_encoded.features.dims(),
        empty_encoded.padding_mask.dims()
    );

    let geometry_encoding = if let Some(geometry_inputs) = geometry_inputs {
        let encoded = model.encode_geometry_prompt(&geometry_inputs.prompt, &visual)?;
        println!(
            "  user prompt: features {:?}, padding mask {:?}",
            encoded.features.dims(),
            encoded.padding_mask.dims()
        );
        Some(encoded)
    } else {
        None
    };

    let prediction_prompt = combine_encoded_prompts(text_encoding, geometry_encoding.as_ref())?;
    if let Some(prediction_prompt) = prediction_prompt {
        let fused = model.encode_fused_prompt(&visual, &prediction_prompt)?;
        println!("fusion stage:");
        println!("  memory shape: {:?}", fused.memory.dims());
        println!("  pos embed shape: {:?}", fused.pos_embed.dims());
        println!("  padding mask shape: {:?}", fused.padding_mask.dims());
        println!(
            "  spatial shapes: {:?}",
            fused.spatial_shapes.to_vec2::<u32>()?
        );
        println!(
            "  level start index: {:?}",
            fused.level_start_index.to_vec1::<u32>()?
        );
        println!("  valid ratios shape: {:?}", fused.valid_ratios.dims());

        let decoder = model.decode_grounding(&fused, &prediction_prompt)?;
        let scores = decode_scores(&decoder)?;
        println!("decoder stage:");
        println!("  queries shape: {:?}", decoder.queries.dims());
        println!("  pred logits shape: {:?}", decoder.pred_logits.dims());
        println!("  pred boxes shape: {:?}", decoder.pred_boxes.dims());
        println!(
            "  pred boxes xyxy shape: {:?}",
            decoder.pred_boxes_xyxy.dims()
        );
        println!("  text detection scores shape: {:?}", scores.dims());
        if let Some(presence_logits) = &decoder.presence_logits {
            println!("  presence logits shape: {:?}", presence_logits.dims());
        }

        let segmentation =
            model.segment_grounding(&visual, &decoder, &fused, &prediction_prompt)?;
        println!("segmentation stage:");
        println!("  mask logits shape: {:?}", segmentation.mask_logits.dims());
        println!(
            "  semantic logits shape: {:?}",
            segmentation.semantic_logits.dims()
        );
        if let Some(presence_logits) = &segmentation.presence_logits {
            println!(
                "  segmentation presence logits shape: {:?}",
                presence_logits.dims()
            );
        }

        let geometry_points = geometry_inputs
            .map(|inputs| inputs.points.as_slice())
            .unwrap_or(&[]);
        let geometry_point_labels = geometry_inputs
            .map(|inputs| inputs.point_labels.as_slice())
            .unwrap_or(&[]);
        let geometry_boxes = geometry_inputs
            .map(|inputs| inputs.boxes.as_slice())
            .unwrap_or(&[]);
        let geometry_box_labels = geometry_inputs
            .map(|inputs| inputs.box_labels.as_slice())
            .unwrap_or(&[]);
        let label = prompt_label(text_prompt, geometry_inputs);
        save_render_outputs(
            image_path,
            image_size,
            output_dir,
            preprocess_mode,
            &label,
            text_prompt,
            &decoder,
            &segmentation,
            &scores,
            geometry_points,
            geometry_point_labels,
            geometry_boxes,
            geometry_box_labels,
            render_style,
        )?;
    }

    Ok(())
}

fn run_batch_jobs(
    model: &sam3::Sam3ImageModel,
    tokenizer_path: Option<&str>,
    source_label: &str,
    jobs: &[BatchJob],
    output_dir: &Path,
    render_style: RenderStyle,
    device: &Device,
) -> Result<()> {
    println!("batch manifest:");
    println!("  source: {source_label}");
    println!("  jobs: {}", jobs.len());
    for (idx, job) in jobs.iter().enumerate() {
        let job_name = job
            .name
            .as_deref()
            .map(sanitize_job_name)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| format!("job-{:02}", idx + 1));
        let job_output_dir = output_dir.join(&job_name);
        println!("running batch job {}/{}: {}", idx + 1, jobs.len(), job_name);
        println!("  image: {}", job.image);
        if let Some(prompt) = job.prompt.as_deref() {
            println!("  text prompt: {prompt}");
        }
        let geometry_inputs = geometry_inputs_from_job(job, device)?;
        let text_encoding = if let Some(prompt) = job.prompt.as_deref() {
            let tokenizer_path = tokenizer_path.ok_or_else(|| {
                E::msg("batch jobs with `prompt` require `--tokenizer <tokenizer.json>`")
            })?;
            Some(run_text_encoder(
                model,
                prompt,
                tokenizer_path,
                model.config().text.context_length,
                device,
            )?)
        } else {
            None
        };
        for preprocess_mode in [PreprocessMode::Exact, PreprocessMode::CropFill] {
            let mode_output_dir = job_output_dir.join(preprocess_mode.as_str());
            println!("  preprocess mode: {}", preprocess_mode.as_str());
            run_vision_and_geometry(
                model,
                &job.image,
                job.smoke_image_size,
                &mode_output_dir,
                preprocess_mode,
                job.prompt.as_deref(),
                text_encoding.as_ref(),
                geometry_inputs.as_ref(),
                render_style,
                device,
            )?;
        }
    }
    Ok(())
}

fn run_batch_manifest(
    model: &sam3::Sam3ImageModel,
    tokenizer_path: Option<&str>,
    manifest_path: &str,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let jobs = load_batch_manifest(manifest_path)?;
    run_batch_jobs(
        model,
        tokenizer_path,
        manifest_path,
        &jobs,
        output_dir,
        RenderStyle::Combined,
        device,
    )
}

fn run_image_predictor_example(
    model: &sam3::Sam3ImageModel,
    tokenizer_path: Option<&str>,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let jobs = image_predictor_example_jobs();
    run_batch_jobs(
        model,
        tokenizer_path,
        "sam3_image_predictor_example.ipynb",
        &jobs,
        output_dir,
        RenderStyle::NotebookImagePredictor,
        device,
    )
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let config = sam3::Config::default();
    let checkpoint_source = args
        .checkpoint
        .as_ref()
        .map(|path| sam3::Sam3CheckpointSource::upstream_pth(resolve_repo_file(path, "sam3.pt")));

    println!("sam3 example");
    println!("device: {device:?}");
    println!(
        "image MVP target: {}x{}",
        config.image.image_size, config.image.image_size
    );
    println!("milestones:");
    for step in sam3::Sam3ImageModel::scaffold_milestones() {
        println!("  - {step}");
    }

    if args.print_config {
        println!("{config:#?}");
    }

    if (args.image.is_some()
        || args.prompt.is_some()
        || !args.points.is_empty()
        || !args.boxes.is_empty()
        || args.parity_bundle.is_some()
        || args.batch_manifest.is_some()
        || args.image_predictor_example)
        && checkpoint_source.is_none()
    {
        bail!("running implemented SAM3 stages currently requires `--checkpoint <sam3.pt>`")
    }
    if (!args.points.is_empty() || !args.boxes.is_empty()) && args.image.is_none() {
        bail!("`--point` and `--box` prompts require `--image` so the geometry encoder has image features")
    }
    if args.parity_bundle.is_some()
        && (args.image.is_some()
            || args.prompt.is_some()
            || args.tokenizer.is_some()
            || !args.points.is_empty()
            || !args.boxes.is_empty()
            || args.batch_manifest.is_some()
            || args.image_predictor_example)
    {
        bail!(
            "`--parity-bundle` uses the exported reference inputs directly; omit `--image`, `--prompt`, `--tokenizer`, `--point`, `--box`, `--batch-manifest`, and `--image-predictor-example`"
        )
    }
    if (args.batch_manifest.is_some() || args.image_predictor_example)
        && (args.image.is_some()
            || args.prompt.is_some()
            || !args.points.is_empty()
            || !args.boxes.is_empty())
    {
        bail!(
            "`--batch-manifest` and `--image-predictor-example` describe their own jobs; omit `--image`, `--prompt`, `--point`, and `--box`"
        )
    }
    if args.batch_manifest.is_some() && args.image_predictor_example {
        bail!("use either `--batch-manifest` or `--image-predictor-example`, not both")
    }

    let model = if let Some(checkpoint) = checkpoint_source.as_ref() {
        let model =
            sam3::Sam3ImageModel::from_checkpoint_source(&config, checkpoint, DType::F32, &device)?;
        println!("checkpoint opened and image-model namespace remap applied");
        Some(model)
    } else {
        None
    };

    if let Some(bundle_path) = args.parity_bundle.as_deref() {
        parity::run(
            model
                .as_ref()
                .context("SAM3 parity mode requires `--checkpoint <sam3.pt>`")?,
            &parity::ParityOptions {
                bundle_path: PathBuf::from(bundle_path),
                output_dir: PathBuf::from(&args.output_dir),
                atol: args.parity_atol,
            },
            &device,
        )?;
        return Ok(());
    }

    if let Some(manifest_path) = args.batch_manifest.as_deref() {
        run_batch_manifest(
            model
                .as_ref()
                .context("SAM3 batch-manifest mode requires `--checkpoint <sam3.pt>`")?,
            args.tokenizer.as_deref(),
            manifest_path,
            Path::new(&args.output_dir),
            &device,
        )?;
        return Ok(());
    }

    if args.image_predictor_example {
        run_image_predictor_example(
            model
                .as_ref()
                .context("SAM3 image-predictor example mode requires `--checkpoint <sam3.pt>`")?,
            args.tokenizer.as_deref(),
            Path::new(&args.output_dir),
            &device,
        )?;
        return Ok(());
    }

    let geometry_inputs = geometry_inputs_from_cli(&args, &device)?;

    let text_encoding = if let Some(prompt) = args.prompt.as_deref() {
        let tokenizer = args.tokenizer.as_deref().ok_or_else(|| {
            E::msg("encoding a SAM3 text prompt requires `--tokenizer <tokenizer.json>`")
        })?;
        Some(run_text_encoder(
            model
                .as_ref()
                .context("SAM3 text stage requires `--checkpoint <sam3.pt>`")?,
            prompt,
            tokenizer,
            config.text.context_length,
            &device,
        )?)
    } else {
        None
    };

    if let Some(image_path) = args.image.as_deref() {
        for preprocess_mode in [PreprocessMode::Exact, PreprocessMode::CropFill] {
            run_vision_and_geometry(
                model
                    .as_ref()
                    .context("SAM3 vision stage requires `--checkpoint <sam3.pt>`")?,
                image_path,
                args.smoke_image_size,
                &Path::new(&args.output_dir).join(preprocess_mode.as_str()),
                preprocess_mode,
                args.prompt.as_deref(),
                text_encoding.as_ref(),
                geometry_inputs.as_ref(),
                RenderStyle::Combined,
                &device,
            )?;
        }
    }

    Ok(())
}
