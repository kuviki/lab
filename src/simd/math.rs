#![allow(non_upper_case_globals)]

/*
An almost-literal transliteration of AVX-optimized sin(), cos(), exp() and log()
functions by LISC, available at http://software-lisc.fbk.eu/avx_mathfun/

which was itself a translation of Simple SSE and SSE2 optimized sin, cos, log,
and exp by Julien Pommier, available at http://gruntthepeon.free.fr/ssemath/ and
licensed under the zlib license (https://en.wikipedia.org/wiki/Zlib_License)
*/

use std::mem;
use std::arch::x86_64::*;

lazy_static! {
    static ref _ps256_1: __m256 = unsafe { _mm256_set1_ps(1.0) };
    static ref _ps256_0p5: __m256 = unsafe { _mm256_set1_ps(0.5) };

    static ref _ps256_min_norm_pos: __m256 = unsafe {
        mem::transmute(_mm256_set1_epi32(0x00800000))
    };
    static ref _ps256_inv_mant_mask: __m256 = unsafe {
        mem::transmute(_mm256_set1_epi32(!0x7f800000))
    };

    static ref _pi32_256_0x7f: __m256i = unsafe{ _mm256_set1_epi32(0x7f) };

    static ref _ps256_cephes_SQRTHF: __m256 = unsafe { _mm256_set1_ps(0.707106781186547524) };
    static ref _ps256_cephes_log_p0: __m256 = unsafe { _mm256_set1_ps(7.0376836292E-2) };
    static ref _ps256_cephes_log_p1: __m256 = unsafe { _mm256_set1_ps(-1.1514610310E-1) };
    static ref _ps256_cephes_log_p2: __m256 = unsafe { _mm256_set1_ps(1.1676998740E-1) };
    static ref _ps256_cephes_log_p3: __m256 = unsafe { _mm256_set1_ps(-1.2420140846E-1) };
    static ref _ps256_cephes_log_p4: __m256 = unsafe { _mm256_set1_ps(1.4249322787E-1) };
    static ref _ps256_cephes_log_p5: __m256 = unsafe { _mm256_set1_ps(-1.6668057665E-1) };
    static ref _ps256_cephes_log_p6: __m256 = unsafe { _mm256_set1_ps(2.0000714765E-1) };
    static ref _ps256_cephes_log_p7: __m256 = unsafe { _mm256_set1_ps(-2.4999993993E-1) };
    static ref _ps256_cephes_log_p8: __m256 = unsafe { _mm256_set1_ps(3.3333331174E-1) };
    static ref _ps256_cephes_log_q1: __m256 = unsafe { _mm256_set1_ps(-2.12194440e-4) };
    static ref _ps256_cephes_log_q2: __m256 = unsafe { _mm256_set1_ps(0.693359375) };

    static ref _ps256_exp_hi: __m256 = unsafe { _mm256_set1_ps(88.3762626647949) };
    static ref _ps256_exp_lo: __m256 = unsafe { _mm256_set1_ps(-88.3762626647949) };

    static ref _ps256_cephes_LOG2EF: __m256 = unsafe { _mm256_set1_ps(1.44269504088896341) };
    static ref _ps256_cephes_exp_C1: __m256 = unsafe { _mm256_set1_ps(0.693359375) };
    static ref _ps256_cephes_exp_C2: __m256 = unsafe { _mm256_set1_ps(-2.12194440e-4) };

    static ref _ps256_cephes_exp_p0: __m256 = unsafe { _mm256_set1_ps(1.9875691500E-4) };
    static ref _ps256_cephes_exp_p1: __m256 = unsafe { _mm256_set1_ps(1.3981999507E-3) };
    static ref _ps256_cephes_exp_p2: __m256 = unsafe { _mm256_set1_ps(8.3334519073E-3) };
    static ref _ps256_cephes_exp_p3: __m256 = unsafe { _mm256_set1_ps(4.1665795894E-2) };
    static ref _ps256_cephes_exp_p4: __m256 = unsafe { _mm256_set1_ps(1.6666665459E-1) };
    static ref _ps256_cephes_exp_p5: __m256 = unsafe { _mm256_set1_ps(5.0000001201E-1) };
}

pub unsafe fn log256_ps(x: __m256) -> __m256 {
    let one = *_ps256_1;

    let invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    let mut x = _mm256_max_ps(x, *_ps256_min_norm_pos);  /* cut off denormalized stuff */

    let mut imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *_ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *_ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, *_pi32_256_0x7f);
    let mut e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    let mask = _mm256_cmp_ps(x, *_ps256_cephes_SQRTHF, _CMP_LT_OS);
    let mut tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    let z = _mm256_mul_ps(x,x);

    let mut y = *_ps256_cephes_log_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    tmp = _mm256_mul_ps(e, *_ps256_cephes_log_q1);
    y = _mm256_add_ps(y, tmp);

    tmp = _mm256_mul_ps(z, *_ps256_0p5);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, *_ps256_cephes_log_q2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
    return x;
}

pub unsafe fn exp256_ps(x: __m256) -> __m256 {
    let one = *_ps256_1;

    let mut x = _mm256_min_ps(x, *_ps256_exp_hi);
    x = _mm256_max_ps(x, *_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    let mut fx = _mm256_mul_ps(x, *_ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, *_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    //imm0 = _mm256_cvttps_epi32(fx);
    //tmp  = _mm256_cvtepi32_ps(imm0);

    let mut tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
    let mut mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, *_ps256_cephes_exp_C1);
    let mut z = _mm256_mul_ps(fx, *_ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x,x);

    let mut y = *_ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *_ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    let mut imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *_pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    let pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

pub unsafe fn powf256_ps(x: __m256, y: __m256) -> __m256 {
    let invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
    let result = exp256_ps(_mm256_mul_ps(y, log256_ps(x)));
    _mm256_or_ps(result, invalid_mask)
}

#[cfg(test)]
mod test {
    use super::{log256_ps, exp256_ps, powf256_ps};
    use std::{f32, mem};
    use std::arch::x86_64::*;

    #[test]
    fn test_log256_ps() {
        let scalar_result: Vec<_> = {
            let vals: [f32; 8] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            vals.iter().copied().map(f32::ln).collect()
        };
        let avx_result: Vec<_> = unsafe {
            let vals = _mm256_set_ps(0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
            let result = log256_ps(vals);
            let result: [f32; 8] = mem::transmute(result);
            result.iter().rev().copied().collect()
        };

        assert_eq!(scalar_result, avx_result)
    }

    #[test]
    fn test_negative_log_returns_nan() {
        let avx_result: Vec<_> = unsafe {
            let vals = _mm256_set_ps(-0.5, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0);
            let result = log256_ps(vals);
            let result: [f32; 8] = mem::transmute(result);
            result.iter().rev().copied().collect()
        };
        assert!(f32::is_nan(avx_result[0]));
        assert!(!f32::is_nan(avx_result[1]));
        assert!(f32::is_nan(avx_result[2]));
        assert!(!f32::is_nan(avx_result[3]));
        assert!(f32::is_nan(avx_result[4]));
        assert!(!f32::is_nan(avx_result[5]));
        assert!(f32::is_nan(avx_result[6]));
        assert!(!f32::is_nan(avx_result[7]));
    }

    #[test]
    fn test_exp256_ps() {
        let scalar_result: Vec<_> = {
            let vals: [f32; 8] = [-1.5, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 10.0];
            vals.iter().copied().map(f32::exp).collect()
        };
        let avx_result: Vec<_> = unsafe {
            let vals = _mm256_set_ps(-1.5, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 10.0);
            let result = exp256_ps(vals);
            let result: [f32; 8] = mem::transmute(result);
            result.iter().rev().copied().collect()
        };

        assert_eq!(scalar_result, avx_result)
    }

    #[test]
    fn test_powf256_ps() {
        let exponent = 4.0;
        let scalar_result: Vec<_> = {
            let vals: [f32; 8] = [0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 10.0];
            vals.iter().map(|&n| n.powf(exponent)).collect()
        };
        let avx_result: Vec<_> = unsafe {
            let vals = _mm256_set_ps(0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 10.0);
            let three = _mm256_set1_ps(exponent);
            let result = powf256_ps(vals, three);
            let result: [f32; 8] = mem::transmute(result);
            result.iter().rev().copied().collect()
        };

        assert_eq!(scalar_result, avx_result)
    }
}
