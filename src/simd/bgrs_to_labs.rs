use super::Lab;
use std::arch::x86_64::*;
use std::iter;
use simd::rgbs_to_labs::{rgbs_to_xyzs, xyzs_to_labs, simd_to_lab_array};

static BLANK_BGR: [u8; 3] = [0u8; 3];

/// Converts a slice of `[u8; 3]` BGR triples to `Lab`s using 256-bit SIMD operations.
///
/// # Panics
/// This function will panic if executed on a non-x86_64 CPU or one without AVX
/// and SSE 4.1 support.
/// ```ignore
/// if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///     lab::simd::bgrs_to_labs(&bgrs);
/// }
/// ```
pub fn bgrs_to_labs(bgrs: &[[u8; 3]]) -> Vec<Lab> {
    let chunks = bgrs.chunks_exact(8);
    let remainder = chunks.remainder();
    let mut vs = chunks.fold(Vec::with_capacity(bgrs.len()), |mut v, bgrs| {
        let labs = unsafe { slice_bgrs_to_slice_labs(bgrs) };
        v.extend_from_slice(&labs);
        v
    });

    // While we could simplify this block by just calling the scalar version
    // of the code on the remainder, there are some variations between scalar
    // and SIMD floating point math (especially on TravisCI for some reason?)
    // and I don't want the trailing N items to be computed by a different
    // algorithm.
    if remainder.len() > 0 {
        let bgrs: Vec<[u8; 3]> = remainder
            .iter()
            .cloned()
            .chain(iter::repeat(BLANK_BGR))
            .take(8)
            .collect();
        let labs = unsafe { slice_bgrs_to_slice_labs(&bgrs) };
        vs.extend_from_slice(&labs[..remainder.len()]);
    }

    vs
}

/// Convert a slice of 8 `[u8; 3]` BGR tripes into an array of 8 `Lab` structs.
///
/// This is the fundamental unit of work that `lab::simd::bgrs_to_labs` performs.
/// If you need to control how to parallelize this work, use this function.
///
/// Only the first 8 elements of the input slice will be converted. The example given
/// is very close to the implementation of `lab::simd::bgrs_to_labs`. Because this
/// library makes no assumptions about how to parallelize work, use this function
/// to add parallelization with Rayon, etc.
///
/// # Example
/// ```
/// # use lab::Lab;
/// # use std::iter;
/// # let bgrs: Vec<[u8; 3]> = vec![];
/// ##[cfg(target_arch = "x86_64")]
/// {
///     if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///         let chunks = bgrs.chunks_exact(8);
///         let remainder = chunks.remainder();
///         // Parallelizing work with Rayon? Do it here, at `.fold()`
///         let mut vs = chunks.fold(Vec::with_capacity(bgrs.len()), |mut v, bgrs| {
///             let labs = lab::simd::bgrs_to_labs_chunk(bgrs);
///             v.extend_from_slice(&labs);
///             v
///         });
///
///         if remainder.len() > 0 {
///             const BLANK_BGR: [u8; 3] = [0u8; 3];
///             let bgrs: Vec<[u8; 3]> =
///                 remainder.iter().cloned().chain(iter::repeat(BLANK_BGR))
///                 .take(8)
///                 .collect();
///
///             let labs = lab::simd::bgrs_to_labs_chunk(&bgrs);
///             vs.extend_from_slice(&labs[..remainder.len()]);
///         }
///     }
/// }
/// ```
///
/// # Panics
/// This function will panic of the input slice has fewer than 8 elements. Consider
/// padding the input slice with blank values and then truncating the result.
///
/// Additionally, it will panic if run on a CPU that does not support x86_64's AVX
/// and SSE 4.1 instructions.
pub fn bgrs_to_labs_chunk(bgrs: &[[u8; 3]]) -> [Lab; 8] {
    unsafe { slice_bgrs_to_slice_labs(bgrs) }
}

unsafe fn slice_bgrs_to_slice_labs(bgrs: &[[u8; 3]]) -> [Lab; 8] {
    let (b, g, r) = bgr_slice_to_simd(bgrs);
    let (x, y, z) = rgbs_to_xyzs(r, g, b);
    let (l, a, b) = xyzs_to_labs(x, y, z);
    simd_to_lab_array(l, a, b)
}

#[inline]
unsafe fn bgr_slice_to_simd(bgrs: &[[u8; 3]]) -> (__m256, __m256, __m256) {
    let b = _mm256_set_ps(
        bgrs[0][0] as f32,
        bgrs[1][0] as f32,
        bgrs[2][0] as f32,
        bgrs[3][0] as f32,
        bgrs[4][0] as f32,
        bgrs[5][0] as f32,
        bgrs[6][0] as f32,
        bgrs[7][0] as f32,
    );
    let g = _mm256_set_ps(
        bgrs[0][1] as f32,
        bgrs[1][1] as f32,
        bgrs[2][1] as f32,
        bgrs[3][1] as f32,
        bgrs[4][1] as f32,
        bgrs[5][1] as f32,
        bgrs[6][1] as f32,
        bgrs[7][1] as f32,
    );
    let r = _mm256_set_ps(
        bgrs[0][2] as f32,
        bgrs[1][2] as f32,
        bgrs[2][2] as f32,
        bgrs[3][2] as f32,
        bgrs[4][2] as f32,
        bgrs[5][2] as f32,
        bgrs[6][2] as f32,
        bgrs[7][2] as f32,
    );
    (b, g, r)
}

// #[cfg(all(target_cpu = "x86_64", target_feature = "avx", target_feature = "sse4.1"))]
#[cfg(test)]
mod test {
    use super::super::super::{bgrs_to_labs, simd};
    use rand;
    use rand::distributions::Standard;
    use rand::Rng;

    lazy_static! {
        static ref BGRS: Vec<[u8; 3]> = {
            let rand_seed = [0u8; 32];
            let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
            rng.sample_iter(&Standard).take(512).collect()
        };
    }

    #[test]
    fn test_simd_bgrs_to_labs() {
        let bgrs = vec![
            [138, 120, 253], // Lab { l: 66.6348, a: 52.260696, b: 14.850557 }
            [22, 20, 25],    // Lab { l: 6.9093895, a: 2.8204322, b: -0.45616925 }
            [181, 81, 63],   // Lab { l: 38.336494, a: 25.586218, b: -55.288517 }
            [102, 132, 21],  // Lab { l: 49.033485, a: -36.959187, b: 7.9363704 }
            [7, 193, 255],   // Lab { l: 81.519325, a: 9.4045105, b: 82.69791 }
            [99, 30, 233],   // Lab { l: 50.865776, a: 74.61989, b: 15.343171 }
            [132, 96, 155],  // Lab { l: 48.260345, a: 29.383003, b: -9.950054 }
            [33, 165, 249],  // Lab { l: 74.29188, a: 21.827251, b: 72.75864 }
        ];

        let labs_non_simd = bgrs_to_labs(&bgrs);
        let labs_simd = simd::bgrs_to_labs(&bgrs);
        assert_eq!(labs_simd, labs_non_simd);
    }

    #[test]
    fn test_simd_bgrs_to_labs_many() {
        let labs_non_simd = bgrs_to_labs(&BGRS);
        let labs_simd = simd::bgrs_to_labs(&BGRS);
        assert_eq!(labs_simd, labs_non_simd);
    }

    #[test]
    fn test_simd_bgrs_to_labs_unsaturated() {
        let bgrs = vec![[253, 120, 138]];
        let labs_non_simd = bgrs_to_labs(&bgrs);
        let labs_simd = simd::bgrs_to_labs(&bgrs);
        assert_eq!(labs_simd, labs_non_simd);
    }
}
