use candle::{IndexOp, Result, Tensor};

pub(crate) fn mask_to_bool_plane(mask: &Tensor, threshold: f32) -> Result<Vec<Vec<bool>>> {
    Ok(tensor_to_mask_probs_2d(mask)?
        .into_iter()
        .map(|row| row.into_iter().map(|value| value >= threshold).collect())
        .collect())
}

pub(crate) fn binary_planes_iou(actual: &[Vec<bool>], expected: &[Vec<bool>]) -> f32 {
    if actual.is_empty() || expected.is_empty() {
        return 0.0;
    }
    let height = actual.len().min(expected.len());
    let width = actual
        .first()
        .map(Vec::len)
        .unwrap_or(0)
        .min(expected.first().map(Vec::len).unwrap_or(0));
    let mut intersection = 0usize;
    let mut union = 0usize;
    for y in 0..height {
        for x in 0..width {
            let a = actual[y][x];
            let b = expected[y][x];
            if a && b {
                intersection += 1;
            }
            if a || b {
                union += 1;
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn tensor_to_mask_probs_2d(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let tensor = match tensor.rank() {
        2 => tensor.clone(),
        3 => tensor.i(0)?,
        4 => tensor.i((0, 0))?,
        rank => candle::bail!("expected mask tensor rank 2/3/4, got {rank}"),
    };
    tensor.to_vec2::<f32>()
}
