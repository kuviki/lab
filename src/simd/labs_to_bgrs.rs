use super::Lab;
use std::arch::x86_64::*;
use std::{f32, iter, mem};
use simd::labs_to_rgbs::{lab_slice_to_simd, labs_to_xyzs, xyzs_to_rgbs};

static BLANK_LAB: Lab = Lab {
    l: f32::NAN,
    a: f32::NAN,
    b: f32::NAN,
};

/// Converts a slice of `Lab`s to `[u8; 3]` BGR triples using 256-bit SIMD operations.
///
/// # Panics
/// This function will panic if executed on a non-x86_64 CPU or one without AVX
/// and SSE 4.1 support.
/// ```ignore
/// if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///     lab::simd::labs_to_bgrs(&labs);
/// }
/// ```
pub fn labs_to_bgrs(labs: &[Lab]) -> Vec<[u8; 3]> {
    let chunks = labs.chunks_exact(8);
    let remainder = chunks.remainder();
    let mut vs = chunks.fold(Vec::with_capacity(labs.len()), |mut v, labs| {
        let bgrs = unsafe { slice_labs_to_slice_bgrs(labs) };
        v.extend_from_slice(&bgrs);
        v
    });

    // While we could simplify this block by just calling the scalar version
    // of the code on the remainder, there are some variations between scalar
    // and SIMD floating point math (especially on TravisCI for some reason?)
    // and I don't want the trailing N items to be computed by a different
    // algorithm.
    if remainder.len() > 0 {
        let labs: Vec<Lab> = remainder
            .iter()
            .cloned()
            .chain(iter::repeat(BLANK_LAB))
            .take(8)
            .collect();

        let bgrs = unsafe { slice_labs_to_slice_bgrs(&labs) };
        vs.extend_from_slice(&bgrs[..remainder.len()]);
    }

    vs
}

/// Convert a slice of 8 `Lab` structs into an array of 8 BGR (`[u8; 3]`) triples.
///
/// This is the fundamental unit of work that `lab::simd::labs_to_bgrs` performs.
/// If you need to control how to parallelize this work, use this function.
///
/// Only the first 8 elements of the input slice will be converted. The example given
/// is very close to the implementation of `lab::simd::labs_to_bgrs`. Because this
/// library makes no assumptions about how to parallelize work, use this function
/// to add parallelization with Rayon, etc.
///
/// # Example
/// ```
/// # use lab::Lab;
/// # use std::{iter, f32};
/// # let labs: Vec<Lab> = {
/// #     let values: &[[f32; 3]] = &[[0.44953918, 0.2343294, 0.9811987], [0.66558355, 0.86746496, 0.6557031], [0.3853534, 0.5447681, 0.563337], [0.5060024, 0.002653122, 0.28564066], [0.112734795, 0.42281234, 0.5662596], [0.61263186, 0.7541826, 0.7710692], [0.35402274, 0.6711668, 0.090500355], [0.09291971, 0.18202633, 0.27621543], [0.74104124, 0.56239027, 0.6807165], [0.19430345, 0.46403062, 0.31903458], [0.9805223, 0.22615737, 0.6665648], [0.61051553, 0.66672426, 0.2612421]];
/// #     values.iter().map(|lab| lab::Lab { l: lab[0], a: lab[1], b: lab[2] }).collect()
/// # };
/// ##[cfg(target_arch = "x86_64")]
/// {
///     if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///         let chunks = labs.chunks_exact(8);
///         let remainder = chunks.remainder();
///         // Parallelizing work with Rayon? Do it here, at `.fold()`
///         let mut vs = chunks.fold(Vec::with_capacity(labs.len()), |mut v, labs| {
///             let bgrs = lab::simd::labs_to_bgrs_chunk(labs);
///             v.extend_from_slice(&bgrs);
///             v
///         });
///
///         if remainder.len() > 0 {
///             const BLANK_LAB: Lab = Lab { l: f32::NAN, a: f32::NAN, b: f32::NAN };
///             let labs: Vec<Lab> =
///                 remainder.iter().cloned().chain(iter::repeat(BLANK_LAB))
///                 .take(8)
///                 .collect();
///
///             let bgrs = lab::simd::labs_to_bgrs_chunk(&labs);
///             vs.extend_from_slice(&bgrs[..remainder.len()]);
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
pub fn labs_to_bgrs_chunk(labs: &[Lab]) -> [[u8; 3]; 8] {
    unsafe { slice_labs_to_slice_bgrs(labs) }
}

#[inline]
unsafe fn slice_labs_to_slice_bgrs(labs: &[Lab]) -> [[u8; 3]; 8] {
    let (l, a, b) = lab_slice_to_simd(labs);
    let (x, y, z) = labs_to_xyzs(l, a, b);
    let (r, g, b) = xyzs_to_rgbs(x, y, z);
    simd_to_bgr_array(b, g, r)
}

#[inline]
unsafe fn simd_to_bgr_array(b: __m256, g: __m256, r: __m256) -> [[u8; 3]; 8] {
    let b: [f32; 8] = mem::transmute(_mm256_round_ps(b, _MM_FROUND_TO_NEAREST_INT));
    let g: [f32; 8] = mem::transmute(_mm256_round_ps(g, _MM_FROUND_TO_NEAREST_INT));
    let r: [f32; 8] = mem::transmute(_mm256_round_ps(r, _MM_FROUND_TO_NEAREST_INT));

    let mut bgrs: [mem::MaybeUninit<[u8; 3]>; 8] = mem::MaybeUninit::uninit().assume_init();
    for (((&b, &g), &r), bgr) in b
        .iter()
        .zip(g.iter())
        .zip(r.iter())
        .rev()
        .zip(bgrs.iter_mut())
    {
        *bgr = mem::MaybeUninit::new([b as u8, g as u8, r as u8]);
    }
    mem::transmute(bgrs)
}

// #[cfg(all(target_cpu = "x86_64", target_feature = "avx", target_feature = "sse4.1"))]
#[cfg(test)]
mod test {
    use super::super::super::{labs_to_bgrs, simd, Lab};
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
    fn test_simd_labs_to_bgrs() {
        let labs = simd::bgrs_to_labs(&BGRS);
        let bgrs = simd::labs_to_bgrs(&labs);
        assert_eq!(bgrs.as_slice(), BGRS.as_slice());
    }

    #[test]
    fn test_simd_labs_to_bgrs_unsaturated() {
        let labs = vec![Lab {
            l: 66.6348,
            a: 52.260696,
            b: 14.850557,
        }];
        let bgrs_non_simd = labs_to_bgrs(&labs);
        let bgrs_simd = simd::labs_to_bgrs(&labs);
        assert_eq!(bgrs_simd, bgrs_non_simd);
    }
}
