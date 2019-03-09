/*
  Rust adaptation of http://gruntthepeon.free.fr/ssemath/sse_mathfun.h
  whose copyright follows:

  Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

use std::arch::x86_64::*;
use std::mem;

unsafe fn exp_ps(x: __m256) -> __m256 {
    let exp_hi = _mm256_set1_ps(88.3762626647949);
    let exp_lo = _mm256_set1_ps(-88.3762626647949);
    let one = _mm256_set1_ps(1.0);

    let mut x = _mm256_min_ps(exp_hi, x);
    x = _mm256_max_ps(exp_lo, x);

    let mut fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341));
    fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5));

    let emm0 = _mm256_cvttps_epi32(fx);
    let mut tmp = _mm256_cvtepi32_ps(emm0);

    let mut mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OQ);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375));
    let mut z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4));
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x, x);

    let mut y = _mm256_set1_ps(1.9875691500E-4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.3981999507E-3));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(8.3334519073E-3));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(4.1665795894E-2));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.6666665459E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(5.0000001201E-1));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    let pow2n = {
        let mut e = _mm256_cvttps_epi32(fx);
        e =  _mm256_add_epi32(e, _mm256_set1_epi32(0x7f));
        e = _mm256_slli_epi32(e, 23);
        _mm256_castsi256_ps(e)
    };
    _mm256_mul_ps(y, pow2n)
}

unsafe fn exp_pd(x: __m256d) -> __m256d {
    let exp_hi = _mm256_set1_pd(88.3762626647949);
    let exp_lo = _mm256_set1_pd(-88.3762626647949);
    let one = _mm256_set1_pd(1.0);

    let mut x = _mm256_min_pd(exp_hi, x);
    x = _mm256_max_pd(exp_lo, x);

    let mut fx = _mm256_mul_pd(x, _mm256_set1_pd(1.44269504088896341));
    fx = _mm256_add_pd(fx, _mm256_set1_pd(0.5));

    let emm0 = _mm256_cvttpd_epi32(fx);
    let mut tmp = _mm256_cvtepi32_pd(emm0);

    let mut mask = _mm256_cmp_pd(tmp, fx, _CMP_GT_OQ);
    mask = _mm256_and_pd(mask, one);
    fx = _mm256_sub_pd(tmp, mask);

    tmp = _mm256_mul_pd(fx, _mm256_set1_pd(0.693359375));
    let mut z = _mm256_mul_pd(fx, _mm256_set1_pd(-2.12194440e-4));
    x = _mm256_sub_pd(x, tmp);
    x = _mm256_sub_pd(x, z);

    z = _mm256_mul_pd(x, x);

    let mut y = _mm256_set1_pd(1.9875691500E-4);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(1.3981999507E-3));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(8.3334519073E-3));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(4.1665795894E-2));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(1.6666665459E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(5.0000001201E-1));
    y = _mm256_mul_pd(y, z);
    y = _mm256_add_pd(y, x);
    y = _mm256_add_pd(y, one);

    let pow2n = {
        let mut e = _mm256_cvttpd_epi32(fx);
        e =  _mm_add_epi32(e, _mm_set1_epi32(0x7f));
        e = _mm_slli_epi32(e, 23);
        let f = _mm_castsi128_ps(e);
        _mm256_cvtps_pd(f)
    };
    _mm256_mul_pd(y, pow2n)
}

// #[test]
// fn test_exp_ps() {
//     const POWERS: [f32; 8] = [0.5432, 1.5432, 2.5432, 3.5432, 4.5432, 5.5432, 6.5432, 7.5432];

//     let res = unsafe {
//         let input = _mm256_loadu_ps(POWERS.as_ptr());
//         let output = exp_ps(input);
//         let mut o: [f32; 8] = mem::uninitialized();
//         _mm256_storeu_ps(o.as_mut_ptr(), output);
//         o
//     };

//     assert_eq!(res.iter().map(|p| *p).collect::<Vec<_>>(), POWERS.iter().map(|p| p.exp()).collect::<Vec<_>>());
// }

// #[test]
// fn test_exp_pd() {
//     const POWERS: [f64; 8] = [0.5432, 1.5432, 2.5432, 3.5432, 4.5432, 5.5432, 6.5432, 7.5432];

//     let res = unsafe {
//         let input = _mm256_loadu_pd(POWERS[..4].as_ptr());
//         let o = exp_pd(input);
//         let mut out: [f64; 4] = mem::uninitialized();
//         _mm256_storeu_pd(out.as_mut_ptr(), o);
//         out
//     };

//     assert_eq!(res.iter().cloned().collect::<Vec<_>>(), POWERS[..4].iter().map(|p| p.exp()).collect::<Vec<_>>());
// }

unsafe fn log_pd(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let invalid_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LE_OQ);

    let (mut x, mut e) = {
        let mut x = _mm_max_ps(_mm256_cvtpd_ps(x), _mm_castsi128_ps(_mm_set1_epi32(0x00800000)));  /* cut off denormalized stuff */

        let mut emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
        /* keep only the fractional part */
        x = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(!0x7f800000)));
        x = _mm_or_ps(x, _mm_set1_ps(0.5));

        emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
        let e = _mm256_cvtepi32_pd(emm0);
        (_mm256_cvtps_pd(x), e)
    };

    e = _mm256_add_pd(e, one);

    let mask = _mm256_cmp_pd(x, _mm256_set1_pd(0.707106781186547524), _CMP_LE_OQ);
    let mut tmp = _mm256_and_pd(x, mask);
    x = _mm256_sub_pd(x, one);
    e = _mm256_sub_pd(e, _mm256_and_pd(one, mask));
    x = _mm256_add_pd(x, tmp);


    let z = _mm256_mul_pd(x,x);

    let mut y = _mm256_set1_pd(7.0376836292E-2);
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(-1.1514610310E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(1.1676998740E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(-1.2420140846E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(1.4249322787E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(-1.6668057665E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(2.0000714765E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(-2.4999993993E-1));
    y = _mm256_mul_pd(y, x);
    y = _mm256_add_pd(y, _mm256_set1_pd(3.3333331174E-1));
    y = _mm256_mul_pd(y, x);

    y = _mm256_mul_pd(y, z);
    

    tmp = _mm256_mul_pd(e, _mm256_set1_pd(-2.12194440e-4));
    y = _mm256_add_pd(y, tmp);


    tmp = _mm256_mul_pd(z, _mm256_set1_pd(0.5));
    y = _mm256_sub_pd(y, tmp);

    tmp = _mm256_mul_pd(e, _mm256_set1_pd(0.693359375));
    x = _mm256_add_pd(x, y);
    x = _mm256_add_pd(x, tmp);
    _mm256_or_pd(x, invalid_mask) // negative arg will be NAN
}

// #[test]
// fn test_log_ps() {
//     const BASE: f32 = 3.0;

//     let res = unsafe {
//         let base = _mm256_set1_ps(BASE);
//         let output = log_ps(base);
//         _mm256_cvtss_f32(output)
//     };

//     assert_eq!(res, BASE.ln());
// }

// #[test]
// fn test_log_pd() {
//     const BASE: f64 = 3.0;

//     let res = unsafe {
//         let base = _mm256_set1_pd(BASE);
//         let output = log_pd(base);
//         _mm256_cvtsd_f64(output)
//     };

//     assert_eq!(res, BASE.ln());
// }

pub unsafe fn power_ps(b: __m256, p: __m256) -> __m256 {
    power(b, p)
}

pub unsafe fn power(b: __m256, p: __m256) -> __m256 {
    let mut bbuf: [f32; 8] = mem::uninitialized();
    _mm256_storeu_ps(bbuf.as_mut_ptr(), b);
    let mut pbuf: [f32; 8] = mem::uninitialized();
    _mm256_storeu_ps(pbuf.as_mut_ptr(), p);

    let lo = _mm256_cvtpd_ps({
        let b = _mm256_cvtps_pd(_mm_loadu_ps(bbuf[..4].as_ptr()));
        let p = _mm256_cvtps_pd(_mm_loadu_ps(pbuf[..4].as_ptr()));
        exp_pd(_mm256_mul_pd(p, log_pd(b)))
    });

    let hi = _mm256_cvtpd_ps({
        let b = _mm256_cvtps_pd(_mm_loadu_ps(bbuf[4..].as_ptr()));
        let p = _mm256_cvtps_pd(_mm_loadu_ps(pbuf[4..].as_ptr()));
        exp_pd(_mm256_mul_pd(p, log_pd(b)))
    });

    let mut buf: [f32; 8] = mem::uninitialized();
    _mm_storeu_ps(buf[..4].as_mut_ptr(), lo);
    _mm_storeu_ps(buf[4..].as_mut_ptr(), hi);
    _mm256_loadu_ps(buf.as_ptr())
}

#[test]
fn test_power_ps() {
    const BASE: f32 = 2.5432;
    const POWER: f32 = 2.5432;

    let res = unsafe {
        let base = _mm256_set1_ps(BASE);
        let power = _mm256_set1_ps(POWER);
        let output = power_ps(base, power);
        _mm256_cvtss_f32(output)
    };

    assert_eq!(res, BASE.powf(POWER));
}

#[test]
fn test_power() {
    const BASE: f32 = 2.5432;
    const POWERS: [f32; 8] = [0.5432, 1.5432, 2.5432, 3.5432, 4.5432, 5.5432, 6.5432, 7.5432];

    let res = unsafe {
        let base = _mm256_set1_ps(BASE);
        let powers = _mm256_loadu_ps(POWERS.as_ptr());
        let mut o: [f32; 8] = mem::uninitialized();
        let out = power_ps(base, powers);
        _mm256_storeu_ps(o.as_mut_ptr(), out);
        o
    };

    assert_eq!(res.iter().cloned().collect::<Vec<_>>(), POWERS.iter().map(|p| BASE.powf(*p)).collect::<Vec<_>>());   
}
