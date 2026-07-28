#[cfg(feature = "sam3-nvtx")]
mod implementation {
    use std::ffi::CString;

    extern "C" {
        fn candle_sam3_nvtx_range_start(message: *const std::ffi::c_char) -> u64;
        fn candle_sam3_nvtx_range_end(range_id: u64);
    }

    #[derive(Debug)]
    pub(crate) struct Range {
        range_id: u64,
    }

    impl Drop for Range {
        fn drop(&mut self) {
            // SAFETY: range_id was returned by the matching wrapper start call.
            unsafe { candle_sam3_nvtx_range_end(self.range_id) }
        }
    }

    pub(crate) fn range(name: &str) -> Range {
        let name = CString::new(name).expect("SAM3 NVTX range contained a nul byte");
        // SAFETY: the wrapper consumes the C string before this function returns.
        let range_id = unsafe { candle_sam3_nvtx_range_start(name.as_ptr()) };
        Range { range_id }
    }
}

#[cfg(not(feature = "sam3-nvtx"))]
mod implementation {
    #[derive(Debug)]
    pub(crate) struct Range;

    #[inline(always)]
    pub(crate) fn range(_name: &str) -> Range {
        Range
    }
}

pub(crate) use implementation::range;
