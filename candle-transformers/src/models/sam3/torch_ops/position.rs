use candle::{DType, Device, IndexOp, Result, Tensor};

use super::tensor::repeat_interleave;

fn sine_position_divisors(device: &Device, len: usize, temperature: f32) -> Result<Tensor> {
    let pair_count = len.div_ceil(2);
    let pair_indices = Tensor::arange(0u32, pair_count as u32, device)?.to_dtype(DType::F32)?;
    let pair_indices = repeat_interleave(&pair_indices, 2, 0)?.narrow(0, 0, len)?;
    pair_indices
        .affine((2.0 * (temperature as f64).ln()) / len as f64, 0.0)?
        .exp()
}

fn axis_positions(device: &Device, extent: usize, normalize: bool, scale: f32) -> Result<Tensor> {
    let positions = Tensor::arange(0u32, extent as u32, device)?
        .to_dtype(DType::F32)?
        .affine(1.0, 1.0)?;
    if normalize {
        positions.affine(scale as f64 / (extent as f64 + 1e-6), 0.0)
    } else {
        Ok(positions)
    }
}

fn interleaved_sin_cos(positions: &Tensor, dim_t: &Tensor) -> Result<Tensor> {
    let extent = positions.dim(0)?;
    let num_pos_feats = dim_t.dim(0)?;
    if num_pos_feats % 2 != 0 {
        candle::bail!(
            "sine position encoding requires an even per-axis width, got {num_pos_feats}"
        )
    }
    let encoded = positions
        .unsqueeze(1)?
        .broadcast_div(&dim_t.unsqueeze(0)?)?
        .reshape((extent, num_pos_feats / 2, 2))?;
    let sin = encoded.i((.., .., 0))?.sin()?;
    let cos = encoded.i((.., .., 1))?.cos()?;
    Tensor::stack(&[&sin, &cos], 2)?.reshape((extent, num_pos_feats))
}

pub(crate) fn get_interleaved_1d_sine_pe(pos_inds: &Tensor, dim: usize) -> Result<Tensor> {
    if dim % 2 != 0 {
        candle::bail!("interleaved sine position encoding requires even dim, got {dim}");
    }
    let device = pos_inds.device();
    let dtype = pos_inds.dtype();
    let flat_pos_inds = pos_inds.flatten_all()?;
    let dim_t = sine_position_divisors(device, dim, 10_000f32)?;
    let pos_embed = flat_pos_inds
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_div(&dim_t.unsqueeze(0)?)?
        .reshape((flat_pos_inds.dim(0)?, dim / 2, 2))?;
    let sin = pos_embed.i((.., .., 0))?.sin()?;
    let cos = pos_embed.i((.., .., 1))?.cos()?;
    let mut output_shape = pos_inds.dims().to_vec();
    output_shape.push(dim);
    Tensor::stack(&[&sin, &cos], 2)?
        .reshape((flat_pos_inds.dim(0)?, dim))?
        .reshape(output_shape)?
        .to_dtype(dtype)
}

pub(crate) fn get_1d_sine_pe(pos_inds: &Tensor, dim: usize) -> Result<Tensor> {
    if dim % 2 != 0 {
        candle::bail!("tracker temporal position encoding requires even dim, got {dim}");
    }
    let device = pos_inds.device();
    let dtype = pos_inds.dtype();
    let pe_dim = dim / 2;
    let dim_t = sine_position_divisors(device, pe_dim, 10_000f32)?;
    let pos_embed = pos_inds
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_div(&dim_t.unsqueeze(0)?)?;
    let sin = pos_embed.sin()?;
    let cos = pos_embed.cos()?;
    Tensor::cat(&[&sin, &cos], 1)?.to_dtype(dtype)
}

pub(crate) fn build_2d_sine_position_encoding_grid(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    num_pos_feats_total: usize,
    height: usize,
    width: usize,
    normalize: bool,
    scale: f32,
    temperature: f32,
) -> Result<Tensor> {
    if num_pos_feats_total % 2 != 0 {
        candle::bail!("2d sine position encoding requires even width, got {num_pos_feats_total}")
    }
    let num_pos_feats = num_pos_feats_total / 2;
    let dim_t = sine_position_divisors(device, num_pos_feats, temperature)?;
    let y_encoded = interleaved_sin_cos(&axis_positions(device, height, normalize, scale)?, &dim_t)?
        .transpose(0, 1)?
        .unsqueeze(2)?
        .broadcast_as((num_pos_feats, height, width))?;
    let x_encoded = interleaved_sin_cos(&axis_positions(device, width, normalize, scale)?, &dim_t)?
        .transpose(0, 1)?
        .unsqueeze(1)?
        .broadcast_as((num_pos_feats, height, width))?;
    let encoding = Tensor::cat(&[&y_encoded, &x_encoded], 0)?.unsqueeze(0)?;
    if batch_size == 1 {
        encoding.to_dtype(dtype)
    } else {
        encoding.repeat((batch_size, 1, 1, 1))?.to_dtype(dtype)
    }
}

pub(crate) fn build_2d_sine_position_encoding(
    feature: &Tensor,
    num_pos_feats_total: usize,
    normalize: bool,
    scale: f32,
    temperature: f32,
) -> Result<Tensor> {
    let (batch_size, channels, height, width) = feature.dims4()?;
    if channels != num_pos_feats_total {
        candle::bail!(
            "2d sine position encoding expected feature width {num_pos_feats_total}, got {channels}"
        )
    }
    build_2d_sine_position_encoding_grid(
        feature.device(),
        feature.dtype(),
        batch_size,
        num_pos_feats_total,
        height,
        width,
        normalize,
        scale,
        temperature,
    )
}
