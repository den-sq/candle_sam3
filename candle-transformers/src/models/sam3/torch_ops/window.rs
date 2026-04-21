use candle::{Result, Tensor};

pub(crate) fn window_partition_nhwc(
    hidden_states: &Tensor,
    window_size: usize,
) -> Result<(Tensor, (usize, usize))> {
    let hidden_states = hidden_states.contiguous()?;
    let (batch_size, height, width, channels) = hidden_states.dims4()?;
    let pad_height = (window_size - height % window_size) % window_size;
    let pad_width = (window_size - width % window_size) % window_size;
    let hidden_states = if pad_height > 0 {
        hidden_states.pad_with_zeros(1, 0, pad_height)?
    } else {
        hidden_states
    };
    let hidden_states = if pad_width > 0 {
        hidden_states.pad_with_zeros(2, 0, pad_width)?
    } else {
        hidden_states
    };
    let padded_height = height + pad_height;
    let padded_width = width + pad_width;
    let windows = hidden_states
        .reshape((
            batch_size,
            padded_height / window_size,
            window_size,
            padded_width / window_size,
            window_size,
            channels,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((
            batch_size * (padded_height / window_size) * (padded_width / window_size),
            window_size,
            window_size,
            channels,
        ))?;
    Ok((windows, (padded_height, padded_width)))
}

pub(crate) fn window_unpartition_nhwc(
    windows: &Tensor,
    window_size: usize,
    padded_hw: (usize, usize),
    original_hw: (usize, usize),
) -> Result<Tensor> {
    let (padded_height, padded_width) = padded_hw;
    let (height, width) = original_hw;
    let num_windows_per_image = padded_height * padded_width / window_size / window_size;
    let batch_size = windows.dim(0)? / num_windows_per_image;
    let hidden_states = windows
        .reshape((
            batch_size,
            padded_height / window_size,
            padded_width / window_size,
            window_size,
            window_size,
            windows.dim(3)?,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((batch_size, padded_height, padded_width, windows.dim(3)?))?;
    let hidden_states = if padded_height > height {
        hidden_states.narrow(1, 0, height)?
    } else {
        hidden_states
    };
    let hidden_states = if padded_width > width {
        hidden_states.narrow(2, 0, width)?
    } else {
        hidden_states
    };
    hidden_states.contiguous()
}
