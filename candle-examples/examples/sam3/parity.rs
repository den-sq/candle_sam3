use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::models::sam3;
use serde::{Deserialize, Serialize};

const REFERENCE_TENSORS_FILE: &str = "reference.safetensors";
const REFERENCE_METADATA_FILE: &str = "reference.json";
const INPUT_IMAGE_KEY: &str = "inputs.image";
const INPUT_IDS_KEY: &str = "inputs.input_ids";
const INPUT_ATTENTION_MASK_KEY: &str = "inputs.attention_mask";

#[derive(Debug, Clone)]
pub struct ParityOptions {
    pub bundle_path: PathBuf,
    pub output_dir: PathBuf,
    pub atol: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParityBundleMetadata {
    #[serde(default = "default_bundle_version")]
    pub bundle_version: usize,
    #[serde(default)]
    pub image_path: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub stage_order: Vec<String>,
}

#[derive(Debug)]
pub struct ParityBundle {
    pub metadata: ParityBundleMetadata,
    tensors: HashMap<String, Tensor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageDiffReport {
    pub stage: String,
    pub expected_shape: Vec<usize>,
    pub actual_shape: Vec<usize>,
    pub max_abs_diff: Option<f32>,
    pub max_abs_diff_flat_index: Option<usize>,
    pub expected_at_max_abs_diff: Option<f32>,
    pub actual_at_max_abs_diff: Option<f32>,
    pub mean_abs_diff: Option<f32>,
    pub rmse: Option<f32>,
    pub pass: bool,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParityReport {
    pub bundle_version: usize,
    pub prompt: Option<String>,
    pub image_path: Option<String>,
    pub image_size: Option<usize>,
    pub atol: f32,
    pub all_passed: bool,
    pub stages: Vec<StageDiffReport>,
}

fn default_bundle_version() -> usize {
    1
}

impl ParityBundle {
    pub fn load(path: &Path) -> Result<Self> {
        let (tensor_path, metadata_path) = resolve_bundle_paths(path);
        let tensors = candle::safetensors::load(&tensor_path, &Device::Cpu).with_context(|| {
            format!(
                "failed to load parity tensor bundle from {}",
                tensor_path.display()
            )
        })?;
        let metadata = if metadata_path.exists() {
            serde_json::from_str::<ParityBundleMetadata>(&fs::read_to_string(&metadata_path)?)
                .with_context(|| {
                    format!(
                        "failed to parse parity metadata from {}",
                        metadata_path.display()
                    )
                })?
        } else {
            ParityBundleMetadata::default()
        };
        let bundle = Self { metadata, tensors };
        bundle.validate()?;
        Ok(bundle)
    }

    fn validate(&self) -> Result<()> {
        for key in [INPUT_IMAGE_KEY, INPUT_IDS_KEY, INPUT_ATTENTION_MASK_KEY] {
            if !self.tensors.contains_key(key) {
                bail!("parity bundle is missing required input tensor `{key}`")
            }
        }
        if !self.tensors.keys().any(|key| !key.starts_with("inputs.")) {
            bail!("parity bundle does not contain any non-input stage tensors")
        }
        Ok(())
    }

    pub fn stage_order(&self) -> Vec<String> {
        if !self.metadata.stage_order.is_empty() {
            return self
                .metadata
                .stage_order
                .iter()
                .filter(|name| !name.starts_with("inputs."))
                .cloned()
                .collect();
        }
        let mut keys = self
            .tensors
            .keys()
            .filter(|name| !name.starts_with("inputs."))
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        keys
    }

    pub fn tensor(&self, key: &str) -> Result<&Tensor> {
        self.tensors
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("parity bundle is missing tensor `{key}`"))
    }
}

pub fn run(model: &sam3::Sam3ImageModel, options: &ParityOptions, device: &Device) -> Result<()> {
    let bundle = ParityBundle::load(&options.bundle_path)?;
    let actual = compute_actual_stages(model, &bundle, device)?;
    let report = build_report(&bundle, &actual, options.atol)?;

    fs::create_dir_all(&options.output_dir)?;
    let report_path = options.output_dir.join("parity_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    let actual_path = options.output_dir.join("actual.safetensors");
    let actual_tensors = actual
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    candle::safetensors::save(&actual_tensors, &actual_path)?;

    println!("parity stage report:");
    if let Some(prompt) = &report.prompt {
        println!("  prompt: {prompt}");
    }
    if let Some(image_path) = &report.image_path {
        println!("  image: {image_path}");
    }
    if let Some(image_size) = report.image_size {
        println!("  image size: {image_size}x{image_size}");
    }
    println!("  absolute tolerance: {}", report.atol);
    for stage in report.stages.iter() {
        match (&stage.note, stage.max_abs_diff) {
            (Some(note), _) => {
                println!("  {}: FAIL ({note})", stage.stage);
            }
            (None, Some(max_abs_diff)) => {
                let status = if stage.pass { "PASS" } else { "FAIL" };
                println!(
                    "  {}: {} (max_abs_diff={:.6}, mean_abs_diff={:.6}, rmse={:.6})",
                    stage.stage,
                    status,
                    max_abs_diff,
                    stage.mean_abs_diff.unwrap_or_default(),
                    stage.rmse.unwrap_or_default()
                );
            }
            (None, None) => {
                println!("  {}: FAIL (no diff statistics produced)", stage.stage);
            }
        }
    }
    println!("  report: {}", report_path.display());
    println!("  actual tensors: {}", actual_path.display());

    if !report.all_passed {
        let failed = report.stages.iter().filter(|stage| !stage.pass).count();
        bail!(
            "sam3 parity check failed in {failed} stage(s); see {}",
            report_path.display()
        )
    }
    Ok(())
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

fn compute_actual_stages(
    model: &sam3::Sam3ImageModel,
    bundle: &ParityBundle,
    device: &Device,
) -> Result<BTreeMap<String, Tensor>> {
    let image = bundle
        .tensor(INPUT_IMAGE_KEY)?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let input_ids = bundle
        .tensor(INPUT_IDS_KEY)?
        .to_device(device)?
        .to_dtype(DType::U32)?;
    let attention_mask = bundle
        .tensor(INPUT_ATTENTION_MASK_KEY)?
        .to_device(device)?
        .to_dtype(DType::U8)?;

    let text = model.encode_text_tokens(&input_ids, &attention_mask)?;
    let expects_block_outputs = bundle
        .tensors
        .keys()
        .any(|key| key.starts_with("vision.block."));
    let debug_block_ids = bundle
        .tensors
        .keys()
        .filter_map(|key| key.strip_prefix("vision.block_debug."))
        .filter_map(|suffix| suffix.split_once('.'))
        .filter_map(|(block_idx, _)| block_idx.parse::<usize>().ok())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let (trunk, block_outputs, debug_tensors) =
        if expects_block_outputs || !debug_block_ids.is_empty() {
            let (trunk, block_outputs, debug_tensors) =
                model.encode_image_trunk_with_debug_blocks(&image, &debug_block_ids)?;
            (trunk, Some(block_outputs), debug_tensors)
        } else {
            (model.encode_image_trunk(&image)?, None, BTreeMap::new())
        };
    let visual = model.encode_image_features(&image)?;
    let geometry = model.encode_geometry_prompt(&sam3::GeometryPrompt::default(), &visual)?;
    let prompt = sam3::EncodedPrompt {
        features: Tensor::cat(&[&text.memory, &geometry.features], 0)?,
        padding_mask: Tensor::cat(&[&text.attention_mask, &geometry.padding_mask], 1)?,
    };
    let fused = model.encode_fused_prompt(&visual, &prompt)?;
    let decoder = model.decode_grounding(&fused, &prompt)?;
    let segmentation = model.segment_grounding(&visual, &decoder, &fused, &prompt)?;

    let mut stages = BTreeMap::new();
    stages.insert(
        "text.input_embeddings".to_owned(),
        text.input_embeddings.clone(),
    );
    stages.insert("text.memory".to_owned(), text.memory.clone());
    if let Some(block_outputs) = block_outputs.as_ref() {
        for (idx, feature_map) in block_outputs.iter().enumerate() {
            stages.insert(
                format!("vision.block.{idx}"),
                feature_map.permute((0, 3, 1, 2))?,
            );
        }
    }
    for (name, feature_map) in debug_tensors.into_iter() {
        stages.insert(name, feature_map.permute((0, 3, 1, 2))?);
    }
    for (idx, feature_map) in trunk.stage_features.iter().enumerate() {
        stages.insert(
            format!("vision.trunk.{idx}"),
            feature_map.permute((0, 3, 1, 2))?,
        );
    }
    for (idx, feature_map) in visual.backbone_fpn.iter().enumerate() {
        stages.insert(format!("vision.backbone_fpn.{idx}"), feature_map.clone());
    }
    stages.insert("fusion.memory".to_owned(), fused.memory.clone());
    stages.insert("fusion.pos_embed".to_owned(), fused.pos_embed.clone());
    stages.insert(
        "fusion.padding_mask".to_owned(),
        fused.padding_mask.to_dtype(DType::U8)?,
    );
    stages.insert(
        "fusion.spatial_shapes".to_owned(),
        fused.spatial_shapes.to_dtype(DType::U32)?,
    );
    stages.insert(
        "fusion.level_start_index".to_owned(),
        fused.level_start_index.to_dtype(DType::U32)?,
    );
    stages.insert(
        "fusion.valid_ratios".to_owned(),
        fused.valid_ratios.to_dtype(DType::F32)?,
    );
    stages.insert(
        "decoder.pred_logits".to_owned(),
        decoder.pred_logits.clone(),
    );
    stages.insert(
        "decoder.pred_boxes_xyxy".to_owned(),
        decoder.pred_boxes_xyxy.clone(),
    );
    if let Some(presence_logits) = &decoder.presence_logits {
        stages.insert(
            "decoder.presence_logits".to_owned(),
            presence_logits.clone(),
        );
    }
    stages.insert(
        "segmentation.mask_logits".to_owned(),
        segmentation.mask_logits.clone(),
    );
    stages.insert(
        "segmentation.semantic_logits".to_owned(),
        segmentation.semantic_logits.clone(),
    );
    if let Some(presence_logits) = &segmentation.presence_logits {
        stages.insert(
            "segmentation.presence_logits".to_owned(),
            presence_logits.clone(),
        );
    }
    Ok(stages)
}

fn build_report(
    bundle: &ParityBundle,
    actual: &BTreeMap<String, Tensor>,
    atol: f32,
) -> Result<ParityReport> {
    let stages = bundle
        .stage_order()
        .into_iter()
        .map(|stage| compare_stage(&stage, bundle, actual, atol))
        .collect::<Result<Vec<_>>>()?;
    let all_passed = stages.iter().all(|stage| stage.pass);
    Ok(ParityReport {
        bundle_version: bundle.metadata.bundle_version,
        prompt: bundle.metadata.prompt.clone(),
        image_path: bundle.metadata.image_path.clone(),
        image_size: bundle.metadata.image_size,
        atol,
        all_passed,
        stages,
    })
}

fn compare_stage(
    stage: &str,
    bundle: &ParityBundle,
    actual: &BTreeMap<String, Tensor>,
    atol: f32,
) -> Result<StageDiffReport> {
    let expected = bundle.tensor(stage)?;
    let expected_shape = expected.dims().to_vec();
    let Some(actual) = actual.get(stage) else {
        return Ok(StageDiffReport {
            stage: stage.to_owned(),
            expected_shape,
            actual_shape: Vec::new(),
            max_abs_diff: None,
            max_abs_diff_flat_index: None,
            expected_at_max_abs_diff: None,
            actual_at_max_abs_diff: None,
            mean_abs_diff: None,
            rmse: None,
            pass: false,
            note: Some("stage missing from Candle output".to_owned()),
        });
    };
    let actual_shape = actual.dims().to_vec();
    if expected_shape != actual_shape {
        return Ok(StageDiffReport {
            stage: stage.to_owned(),
            expected_shape,
            actual_shape,
            max_abs_diff: None,
            max_abs_diff_flat_index: None,
            expected_at_max_abs_diff: None,
            actual_at_max_abs_diff: None,
            mean_abs_diff: None,
            rmse: None,
            pass: false,
            note: Some("shape mismatch".to_owned()),
        });
    }

    let expected = expected.to_dtype(DType::F32)?.flatten_all()?;
    let actual = actual.to_dtype(DType::F32)?.flatten_all()?;
    let expected = expected.to_vec1::<f32>()?;
    let actual = actual.to_vec1::<f32>()?;
    let len = expected.len();
    if len == 0 {
        return Ok(StageDiffReport {
            stage: stage.to_owned(),
            expected_shape,
            actual_shape,
            max_abs_diff: Some(0.0),
            max_abs_diff_flat_index: Some(0),
            expected_at_max_abs_diff: Some(0.0),
            actual_at_max_abs_diff: Some(0.0),
            mean_abs_diff: Some(0.0),
            rmse: Some(0.0),
            pass: true,
            note: None,
        });
    }

    let mut max_abs_diff = 0f32;
    let mut max_abs_diff_flat_index = 0usize;
    let mut expected_at_max_abs_diff = 0f32;
    let mut actual_at_max_abs_diff = 0f32;
    let mut sum_abs_diff = 0f64;
    let mut sum_sq_diff = 0f64;
    for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let abs_diff = if expected.is_nan() || actual.is_nan() {
            f32::INFINITY
        } else {
            (actual - expected).abs()
        };
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            max_abs_diff_flat_index = index;
            expected_at_max_abs_diff = *expected;
            actual_at_max_abs_diff = *actual;
        }
        sum_abs_diff += abs_diff as f64;
        sum_sq_diff += (abs_diff as f64) * (abs_diff as f64);
    }
    let mean_abs_diff = (sum_abs_diff / len as f64) as f32;
    let rmse = (sum_sq_diff / len as f64).sqrt() as f32;

    Ok(StageDiffReport {
        stage: stage.to_owned(),
        expected_shape,
        actual_shape,
        max_abs_diff: Some(max_abs_diff),
        max_abs_diff_flat_index: Some(max_abs_diff_flat_index),
        expected_at_max_abs_diff: Some(expected_at_max_abs_diff),
        actual_at_max_abs_diff: Some(actual_at_max_abs_diff),
        mean_abs_diff: Some(mean_abs_diff),
        rmse: Some(rmse),
        pass: max_abs_diff <= atol,
        note: None,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle::{Device, Tensor};

    use super::{compare_stage, ParityBundle, ParityBundleMetadata};
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn compare_stage_reports_small_float_diffs() -> Result<()> {
        let device = Device::Cpu;
        let bundle = ParityBundle {
            metadata: ParityBundleMetadata::default(),
            tensors: HashMap::from([(
                "decoder.pred_logits".to_owned(),
                Tensor::from_vec(vec![1f32, 2f32], (1, 2), &device)?,
            )]),
        };
        let actual = BTreeMap::from([(
            "decoder.pred_logits".to_owned(),
            Tensor::from_vec(vec![1f32, 2.0005f32], (1, 2), &device)?,
        )]);

        let report = compare_stage("decoder.pred_logits", &bundle, &actual, 1e-3)?;
        assert!(report.pass);
        assert_eq!(report.expected_shape, vec![1, 2]);
        assert!(report.max_abs_diff.unwrap() > 0.0);
        Ok(())
    }

    #[test]
    fn compare_stage_fails_on_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let bundle = ParityBundle {
            metadata: ParityBundleMetadata::default(),
            tensors: HashMap::from([(
                "segmentation.mask_logits".to_owned(),
                Tensor::zeros((1, 2, 3), candle::DType::F32, &device)?,
            )]),
        };
        let actual = BTreeMap::from([(
            "segmentation.mask_logits".to_owned(),
            Tensor::zeros((1, 2, 4), candle::DType::F32, &device)?,
        )]);

        let report = compare_stage("segmentation.mask_logits", &bundle, &actual, 1e-4)?;
        assert!(!report.pass);
        assert_eq!(report.note.as_deref(), Some("shape mismatch"));
        Ok(())
    }
}
