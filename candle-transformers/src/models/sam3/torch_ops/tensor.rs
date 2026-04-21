use candle::{Result, Tensor};

pub(crate) fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let xs = xs.unsqueeze(dim + 1)?;
    let mut dims = xs.dims().to_vec();
    dims[dim + 1] = repeats;
    xs.broadcast_as(dims)?.flatten(dim, dim + 1)
}
