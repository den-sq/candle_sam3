use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::models::sam3;
use serde::{Deserialize, Serialize};

use crate::interactive::InteractiveReplayStep;

const REFERENCE_TENSORS_FILE: &str = "reference.safetensors";
const REFERENCE_METADATA_FILE: &str = "reference.json";

#[derive(Debug, Clone, Deserialize)]
pub struct InteractiveReferenceMetadata {
    #[serde(default = "default_bundle_version")]
    pub bundle_version: usize,
    pub image_path: String,
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub preprocess_mode: Option<String>,
    #[serde(default)]
    pub replay_script_path: Option<String>,
    #[serde(default)]
    pub steps: Vec<InteractiveReferenceStepMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InteractiveReferenceStepMetadata {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub step_points_xy_normalized: Vec<Vec<f32>>,
    #[serde(default)]
    pub step_point_labels: Vec<u32>,
    #[serde(default)]
    pub accumulated_points_xy_normalized: Vec<Vec<f32>>,
    #[serde(default)]
    pub accumulated_point_labels: Vec<u32>,
}

#[derive(Debug)]
pub struct InteractiveReferenceBundle {
    pub metadata: InteractiveReferenceMetadata,
    tensors: HashMap<String, Tensor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InteractiveComparisonStageReport {
    pub stage: String,
    pub expected_shape: Vec<usize>,
    pub actual_shape: Vec<usize>,
    pub max_abs_diff: Option<f32>,
    pub mean_abs_diff: Option<f32>,
    pub rmse: Option<f32>,
    pub pass: bool,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InteractiveComparisonEntry {
    pub iteration_index: usize,
    pub step_name: String,
    pub score_abs_diff: f32,
    pub reference_best_score: f32,
    pub candle_best_score: f32,
    pub reference_best_box_xyxy: Vec<f32>,
    pub candle_best_box_xyxy: Vec<f32>,
    pub box_l1_mean_abs_diff: f32,
    pub box_iou: f32,
    pub mask_mean_abs_diff: f32,
    pub mask_iou_threshold_0_5: f32,
    pub stages: Vec<InteractiveComparisonStageReport>,
    pub all_stages_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct InteractiveComparisonReport {
    pub bundle_version: usize,
    pub image_path: String,
    pub image_size: usize,
    pub preprocess_mode: String,
    pub replay_script_path: Option<String>,
    pub atol: f32,
    pub all_passed: bool,
    pub steps: Vec<InteractiveComparisonEntry>,
}

#[derive(Debug)]
struct CandleInteractiveStepOutputs {
    geometry_features: Tensor,
    geometry_padding_mask: Tensor,
    fusion_memory: Tensor,
    decoder_pred_logits: Tensor,
    decoder_pred_boxes_xyxy: Tensor,
    decoder_presence_logits: Option<Tensor>,
    segmentation_mask_logits: Tensor,
    scores: Tensor,
}

fn default_bundle_version() -> usize {
    1
}

impl InteractiveReferenceBundle {
    pub fn load(path: &Path) -> Result<Self> {
        let (tensor_path, metadata_path) = resolve_bundle_paths(path);
        let tensors = candle::safetensors::load(&tensor_path, &Device::Cpu).with_context(|| {
            format!(
                "failed to load interactive reference tensor bundle from {}",
                tensor_path.display()
            )
        })?;
        let metadata = serde_json::from_str::<InteractiveReferenceMetadata>(
            &fs::read_to_string(&metadata_path).with_context(|| {
                format!(
                    "failed to read interactive reference metadata from {}",
                    metadata_path.display()
                )
            })?,
        )
        .with_context(|| {
            format!(
                "failed to parse interactive reference metadata from {}",
                metadata_path.display()
            )
        })?;
        let bundle = Self { metadata, tensors };
        bundle.validate()?;
        Ok(bundle)
    }

    fn validate(&self) -> Result<()> {
        if self.metadata.steps.is_empty() {
            bail!("interactive reference bundle does not contain any steps")
        }
        for step_idx in 0..self.metadata.steps.len() {
            for key in [
                format!("step.{step_idx}.geometry.features"),
                format!("step.{step_idx}.geometry.padding_mask"),
                format!("step.{step_idx}.fusion.memory"),
                format!("step.{step_idx}.decoder.pred_logits"),
                format!("step.{step_idx}.decoder.pred_boxes_xyxy"),
                format!("step.{step_idx}.segmentation.mask_logits"),
            ] {
                if !self.tensors.contains_key(&key) {
                    bail!("interactive reference bundle is missing required tensor `{key}`")
                }
            }
        }
        Ok(())
    }

    pub fn tensor(&self, key: &str) -> Result<&Tensor> {
        self.tensors.get(key).ok_or_else(|| {
            anyhow::anyhow!("interactive reference bundle is missing tensor `{key}`")
        })
    }

    pub fn tensor_opt(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }
}

fn resolve_bundle_paths(path: &Path) -> (PathBuf, PathBuf) {
    if path.is_dir() {
        (
            path.join(REFERENCE_TENSORS_FILE),
            path.join(REFERENCE_METADATA_FILE),
        )
    } else {
        let tensor_path = path.to_path_buf();
        let metadata_path = tensor_path.with_extension("json");
        (tensor_path, metadata_path)
    }
}

fn point_args_from_pairs(points: &[(f32, f32)]) -> Vec<crate::PointArg> {
    points
        .iter()
        .map(|(x, y)| crate::PointArg { x: *x, y: *y })
        .collect()
}

fn interactive_replay_steps_from_metadata(
    metadata: &[InteractiveReferenceStepMetadata],
) -> Result<Vec<InteractiveReplayStep>> {
    metadata
        .iter()
        .enumerate()
        .map(|(idx, step)| {
            let points = step
                .step_points_xy_normalized
                .iter()
                .map(|point| -> Result<(f32, f32)> {
                    if point.len() != 2 {
                        bail!(
                            "interactive reference step {} expected point [x, y], got {} values",
                            idx,
                            point.len()
                        )
                    }
                    Ok((point[0], point[1]))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(InteractiveReplayStep {
                name: step.name.clone(),
                points,
                point_labels: step.step_point_labels.clone(),
            })
        })
        .collect()
}

fn compare_tensor(
    stage: &str,
    expected: &Tensor,
    actual: &Tensor,
    atol: f32,
) -> Result<InteractiveComparisonStageReport> {
    let expected_shape = expected.dims().to_vec();
    let actual_shape = actual.dims().to_vec();
    if expected_shape != actual_shape {
        return Ok(InteractiveComparisonStageReport {
            stage: stage.to_string(),
            expected_shape,
            actual_shape,
            max_abs_diff: None,
            mean_abs_diff: None,
            rmse: None,
            pass: false,
            note: Some("shape mismatch".to_string()),
        });
    }

    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if expected.is_empty() {
        return Ok(InteractiveComparisonStageReport {
            stage: stage.to_string(),
            expected_shape,
            actual_shape,
            max_abs_diff: Some(0.0),
            mean_abs_diff: Some(0.0),
            rmse: Some(0.0),
            pass: true,
            note: None,
        });
    }

    let mut max_abs_diff = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (lhs, rhs) in expected.iter().zip(actual.iter()) {
        let diff = if lhs.is_nan() || rhs.is_nan() {
            f32::INFINITY
        } else {
            (lhs - rhs).abs()
        };
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs += diff as f64;
        sum_sq += (diff as f64) * (diff as f64);
    }
    let len = expected.len() as f64;
    let mean_abs_diff = (sum_abs / len) as f32;
    let rmse = (sum_sq / len).sqrt() as f32;
    Ok(InteractiveComparisonStageReport {
        stage: stage.to_string(),
        expected_shape,
        actual_shape,
        max_abs_diff: Some(max_abs_diff),
        mean_abs_diff: Some(mean_abs_diff),
        rmse: Some(rmse),
        pass: max_abs_diff <= atol,
        note: None,
    })
}

fn run_candle_interactive_step(
    model: &sam3::Sam3ImageModel,
    image: &Tensor,
    points: &[(f32, f32)],
    point_labels: &[u32],
    device: &Device,
) -> Result<CandleInteractiveStepOutputs> {
    let geometry_inputs = crate::build_geometry_prompt_from_parts(
        &point_args_from_pairs(points),
        point_labels,
        &[],
        &[],
        device,
    )?
    .context("interactive comparison step expected non-empty geometry prompt")?;
    let visual = model.encode_image_features(image)?;
    let geometry = model.encode_geometry_prompt(&geometry_inputs.prompt, &visual)?;
    let fused = model.encode_fused_prompt(&visual, &geometry)?;
    let decoder = model.decode_grounding(&fused, &geometry)?;
    let scores = crate::decode_scores(&decoder)?;
    let segmentation = model.segment_grounding(&visual, &decoder, &fused, &geometry)?;
    Ok(CandleInteractiveStepOutputs {
        geometry_features: geometry.features,
        geometry_padding_mask: geometry.padding_mask.to_dtype(DType::U8)?,
        fusion_memory: fused.memory,
        decoder_pred_logits: decoder.pred_logits,
        decoder_pred_boxes_xyxy: decoder.pred_boxes_xyxy,
        decoder_presence_logits: decoder.presence_logits,
        segmentation_mask_logits: segmentation.mask_logits,
        scores,
    })
}

pub fn run_interactive_reference_comparison(
    model: &sam3::Sam3ImageModel,
    bundle_path: &str,
    output_dir: &Path,
    device: &Device,
    atol: f32,
) -> Result<()> {
    println!("loading interactive reference bundle from {bundle_path}");
    let bundle = InteractiveReferenceBundle::load(Path::new(bundle_path))?;
    fs::create_dir_all(output_dir)?;
    let image_size = bundle
        .metadata
        .image_size
        .unwrap_or(model.config().image.image_size);
    let preprocess_mode =
        crate::PreprocessMode::from_bundle_metadata(bundle.metadata.preprocess_mode.as_deref())?;
    if preprocess_mode != crate::PreprocessMode::Exact {
        bail!(
            "interactive reference comparison currently expects exact preprocessing, got `{}`",
            preprocess_mode.as_str()
        )
    }

    println!(
        "preprocessing reference image {}",
        bundle.metadata.image_path
    );
    let image = crate::preprocess_image_path_exact(&bundle.metadata.image_path, model, device)?;
    let replay_steps = interactive_replay_steps_from_metadata(&bundle.metadata.steps)?;
    println!(
        "loaded {} interactive replay step(s) for direct comparison",
        replay_steps.len()
    );

    let mut entries = Vec::with_capacity(bundle.metadata.steps.len());
    for (step_idx, step) in bundle.metadata.steps.iter().enumerate() {
        println!("comparing interactive step {step_idx}");
        let accumulated_points = step
            .accumulated_points_xy_normalized
            .iter()
            .map(|point| -> Result<(f32, f32)> {
                if point.len() != 2 {
                    bail!(
                        "interactive reference step {} accumulated point expected [x, y], got {} values",
                        step_idx,
                        point.len()
                    )
                }
                Ok((point[0], point[1]))
            })
            .collect::<Result<Vec<_>>>()?;
        let candle = run_candle_interactive_step(
            model,
            &image,
            &accumulated_points,
            &step.accumulated_point_labels,
            device,
        )?;

        let stages = vec![
            compare_tensor(
                "geometry.features",
                bundle.tensor(&format!("step.{step_idx}.geometry.features"))?,
                &candle.geometry_features,
                atol,
            )?,
            compare_tensor(
                "geometry.padding_mask",
                bundle.tensor(&format!("step.{step_idx}.geometry.padding_mask"))?,
                &candle.geometry_padding_mask,
                atol,
            )?,
            compare_tensor(
                "fusion.memory",
                bundle.tensor(&format!("step.{step_idx}.fusion.memory"))?,
                &candle.fusion_memory,
                atol,
            )?,
            compare_tensor(
                "decoder.pred_logits",
                bundle.tensor(&format!("step.{step_idx}.decoder.pred_logits"))?,
                &candle.decoder_pred_logits,
                atol,
            )?,
            compare_tensor(
                "decoder.pred_boxes_xyxy",
                bundle.tensor(&format!("step.{step_idx}.decoder.pred_boxes_xyxy"))?,
                &candle.decoder_pred_boxes_xyxy,
                atol,
            )?,
            compare_tensor(
                "segmentation.mask_logits",
                bundle.tensor(&format!("step.{step_idx}.segmentation.mask_logits"))?,
                &candle.segmentation_mask_logits,
                atol,
            )?,
        ];
        let mut stages = stages;
        if let Some(reference_presence_logits) =
            bundle.tensor_opt(&format!("step.{step_idx}.decoder.presence_logits"))
        {
            let actual_presence_logits = candle
                .decoder_presence_logits
                .as_ref()
                .context("Candle interactive step did not produce decoder presence logits")?;
            stages.push(compare_tensor(
                "decoder.presence_logits",
                reference_presence_logits,
                actual_presence_logits,
                atol,
            )?);
        }

        let reference_scores = crate::decode_scores_from_tensors(
            bundle.tensor(&format!("step.{step_idx}.decoder.pred_logits"))?,
            bundle.tensor_opt(&format!("step.{step_idx}.decoder.presence_logits")),
        )?;
        let reference_selected = crate::select_prediction_from_xyxy_tensors(
            &bundle.metadata.image_path,
            image_size,
            preprocess_mode,
            bundle.tensor(&format!("step.{step_idx}.decoder.pred_boxes_xyxy"))?,
            bundle.tensor(&format!("step.{step_idx}.segmentation.mask_logits"))?,
            &reference_scores,
        )?;
        let candle_selected = crate::select_prediction_from_xyxy_tensors(
            &bundle.metadata.image_path,
            image_size,
            preprocess_mode,
            &candle.decoder_pred_boxes_xyxy,
            &candle.segmentation_mask_logits,
            &candle.scores,
        )?;

        let all_stages_passed = stages.iter().all(|stage| stage.pass);
        let entry = InteractiveComparisonEntry {
            iteration_index: step_idx,
            step_name: step
                .name
                .clone()
                .unwrap_or_else(|| format!("step_{step_idx:02}")),
            score_abs_diff: (reference_selected.best_score - candle_selected.best_score).abs(),
            reference_best_score: reference_selected.best_score,
            candle_best_score: candle_selected.best_score,
            reference_best_box_xyxy: reference_selected.best_box_xyxy.clone(),
            candle_best_box_xyxy: candle_selected.best_box_xyxy.clone(),
            box_l1_mean_abs_diff: crate::mean_abs_box_diff(
                &reference_selected.best_box_xyxy,
                &candle_selected.best_box_xyxy,
            ),
            box_iou: crate::box_iou(
                &reference_selected.best_box_xyxy,
                &candle_selected.best_box_xyxy,
            ),
            mask_mean_abs_diff: crate::mask_mean_abs_diff(
                &reference_selected.mask_probs,
                &candle_selected.mask_probs,
                None,
            )?,
            mask_iou_threshold_0_5: crate::mask_iou_at_threshold(
                &reference_selected.mask_probs,
                &candle_selected.mask_probs,
                0.5,
                None,
            )?,
            stages,
            all_stages_passed,
        };
        let status = if entry.all_stages_passed {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  step {} ({}): {} score_diff={:.6} box_iou={:.6} mask_mae={:.6} mask_iou@0.5={:.6}",
            entry.iteration_index,
            entry.step_name,
            status,
            entry.score_abs_diff,
            entry.box_iou,
            entry.mask_mean_abs_diff,
            entry.mask_iou_threshold_0_5
        );
        if let Some(first_fail) = entry.stages.iter().find(|stage| !stage.pass) {
            println!(
                "    first failing stage: {}{}",
                first_fail.stage,
                first_fail
                    .note
                    .as_deref()
                    .map(|note| format!(" ({note})"))
                    .unwrap_or_default()
            );
        }
        entries.push(entry);

        let partial_report = InteractiveComparisonReport {
            bundle_version: bundle.metadata.bundle_version,
            image_path: bundle.metadata.image_path.clone(),
            image_size,
            preprocess_mode: preprocess_mode.as_str().to_string(),
            replay_script_path: bundle.metadata.replay_script_path.clone(),
            atol,
            all_passed: entries.iter().all(|current| current.all_stages_passed),
            steps: entries.clone(),
        };
        let report_path = output_dir.join("interactive_comparison_report.json");
        fs::write(&report_path, serde_json::to_string_pretty(&partial_report)?)?;
        if !partial_report.all_passed {
            bail!(
                "interactive replay comparison failed at step {}; see {}",
                step_idx,
                report_path.display()
            );
        }
    }

    let report = InteractiveComparisonReport {
        bundle_version: bundle.metadata.bundle_version,
        image_path: bundle.metadata.image_path.clone(),
        image_size,
        preprocess_mode: preprocess_mode.as_str().to_string(),
        replay_script_path: bundle.metadata.replay_script_path.clone(),
        atol,
        all_passed: entries.iter().all(|entry| entry.all_stages_passed),
        steps: entries,
    };

    let report_path = output_dir.join("interactive_comparison_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

    println!("interactive replay comparison:");
    println!("  image: {}", report.image_path);
    println!("  image size: {}x{}", report.image_size, report.image_size);
    println!("  preprocess mode: {}", report.preprocess_mode);
    println!("  absolute tolerance: {}", report.atol);
    for entry in &report.steps {
        let status = if entry.all_stages_passed {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  step {} ({}): {} score_diff={:.6} box_iou={:.6} mask_mae={:.6} mask_iou@0.5={:.6}",
            entry.iteration_index,
            entry.step_name,
            status,
            entry.score_abs_diff,
            entry.box_iou,
            entry.mask_mean_abs_diff,
            entry.mask_iou_threshold_0_5
        );
    }
    println!("  report: {}", report_path.display());

    if !report.all_passed {
        bail!(
            "interactive replay comparison failed; see {}",
            report_path.display()
        )
    }
    Ok(())
}
