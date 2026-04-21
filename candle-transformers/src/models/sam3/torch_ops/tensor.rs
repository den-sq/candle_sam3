use candle::{DType, IndexOp, Result, Tensor};

pub(crate) fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let xs = xs.unsqueeze(dim + 1)?;
    let mut dims = xs.dims().to_vec();
    dims[dim + 1] = repeats;
    xs.broadcast_as(dims)?.flatten(dim, dim + 1)
}

pub(crate) fn first_scalar_f32(xs: &Tensor) -> Result<f32> {
    let flat = xs.to_dtype(DType::F32)?.flatten_all()?;
    if flat.elem_count() == 0 {
        return Ok(0.0);
    }
    flat.i(0)?.to_scalar::<f32>()
}

pub(crate) fn flatten_all_contiguous(xs: &Tensor) -> Result<Tensor> {
    xs.flatten_all()?.contiguous()
}
