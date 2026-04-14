use anyhow::Result;
use candle::{DType, Tensor};

#[derive(Debug, Clone)]
pub struct TensorComparisonReport {
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

pub fn compare_tensors(
    expected: &Tensor,
    actual: Option<&Tensor>,
    atol: f32,
    missing_note: &str,
) -> Result<TensorComparisonReport> {
    let expected_shape = expected.dims().to_vec();
    let Some(actual) = actual else {
        return Ok(TensorComparisonReport {
            expected_shape,
            actual_shape: Vec::new(),
            max_abs_diff: None,
            max_abs_diff_flat_index: None,
            expected_at_max_abs_diff: None,
            actual_at_max_abs_diff: None,
            mean_abs_diff: None,
            rmse: None,
            pass: false,
            note: Some(missing_note.to_owned()),
        });
    };
    let actual_shape = actual.dims().to_vec();
    if expected_shape != actual_shape {
        return Ok(TensorComparisonReport {
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

    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let len = expected.len();
    if len == 0 {
        return Ok(TensorComparisonReport {
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

    Ok(TensorComparisonReport {
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

    use super::compare_tensors;

    #[test]
    fn compare_tensors_reports_small_float_diffs() -> Result<()> {
        let device = Device::Cpu;
        let expected = Tensor::from_vec(vec![1f32, 2f32], (1, 2), &device)?;
        let actual = Tensor::from_vec(vec![1f32, 2.0005f32], (1, 2), &device)?;

        let report = compare_tensors(&expected, Some(&actual), 1e-3, "missing")?;
        assert!(report.pass);
        assert_eq!(report.expected_shape, vec![1, 2]);
        assert!(report.max_abs_diff.unwrap() > 0.0);
        Ok(())
    }

    #[test]
    fn compare_tensors_fails_on_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let expected = Tensor::zeros((1, 2, 3), candle::DType::F32, &device)?;
        let actual = Tensor::zeros((1, 2, 4), candle::DType::F32, &device)?;

        let report = compare_tensors(&expected, Some(&actual), 1e-4, "missing")?;
        assert!(!report.pass);
        assert_eq!(report.note.as_deref(), Some("shape mismatch"));
        Ok(())
    }

    #[test]
    fn compare_tensors_reports_missing_actual_tensor() -> Result<()> {
        let device = Device::Cpu;
        let expected = Tensor::zeros((1, 2), candle::DType::F32, &device)?;

        let report = compare_tensors(&expected, None, 1e-4, "missing stage")?;
        assert!(!report.pass);
        assert_eq!(report.note.as_deref(), Some("missing stage"));
        assert_eq!(report.actual_shape, Vec::<usize>::new());
        Ok(())
    }
}
