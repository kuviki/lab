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

#[test]
fn test_exp_ps() {
    const POWER: f32 = 1.5;

    let res = unsafe {
        let input = _mm256_set1_ps(POWER);
        let output = exp_ps(input);
        _mm256_cvtss_f32(output)
    };

    assert_eq!(res, POWER.exp());
}

unsafe fn log_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);

    let mut x = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000)));  /* cut off denormalized stuff */

    let mut emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    /* keep only the fractional part */
    x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(!0x7f800000)));
    x = _mm256_or_ps(x, _mm256_set1_ps(0.5));

    emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
    let mut e = _mm256_cvtepi32_ps(emm0);

    e = _mm256_add_ps(e, one);

    let mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524), _CMP_LE_OQ);
    let mut tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);


    let z = _mm256_mul_ps(x,x);

    let mut y = _mm256_set1_ps(7.0376836292E-2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.1514610310E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.1676998740E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.2420140846E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.4249322787E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.6668057665E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(2.0000714765E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-2.4999993993E-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(3.3333331174E-1));
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);
    

    tmp = _mm256_mul_ps(e, _mm256_set1_ps(-2.12194440e-4));
    y = _mm256_add_ps(y, tmp);


    tmp = _mm256_mul_ps(z, _mm256_set1_ps(0.5));
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, _mm256_set1_ps(0.693359375));
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    _mm256_or_ps(x, invalid_mask) // negative arg will be NAN
}

#[test]
fn test_log_ps() {
    const BASE: f32 = 3.0;

    let res = unsafe {
        let base = _mm256_set1_ps(BASE);
        let output = log_ps(base);
        _mm256_cvtss_f32(output)
    };

    assert_eq!(res, BASE.ln());
}

pub unsafe fn power_ps(b: __m256, p: __m256) -> __m256 {
    exp_ps(_mm256_mul_ps(p, log_ps(b)))
}

#[test]
fn test_power_ps() {
    const BASE: f32 = 2.4;
    const POWER: f32 = 2.4;

    let res = unsafe {
        let base = _mm256_set1_ps(BASE);
        let power = _mm256_set1_ps(POWER);
        let output = power_ps(base, power);
        _mm256_cvtss_f32(output)
    };

    assert_eq!(res, BASE.powf(POWER));
}
