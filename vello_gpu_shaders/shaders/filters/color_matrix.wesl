// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// A 4x5 color matrix (feColorMatrix) applied to a premultiplied color: the
// matrix works on unpremultiplied red, green, blue and alpha, and each output
// channel is clamped to [0, 1]. The 20 coefficients are packed as f16 pairs
// after the header: texel0.yzw, texel1.xyzw, texel2.xyz (row-major, each row
// R, G, B, A, offset). Keep in sync with `GpuColorMatrix` in `filter.rs`.
// An IEEE 754 half (the low 16 bits of `h`) as an f32, decoded by hand:
// `unpack2x16float` needs a float16 capability not every target has.
fn half_to_f32(h: u32) -> f32 {
    let exponent = (h >> 10u) & 0x1fu;
    let mantissa = f32(h & 0x3ffu);
    var magnitude = mantissa * 5.9604645e-8; // subnormal: m * 2^-24
    if exponent != 0u {
        magnitude = (1.0 + mantissa / 1024.0) * exp2(f32(exponent) - 15.0);
    }
    return select(magnitude, -magnitude, (h & 0x8000u) != 0u);
}

fn unpack_halves(packed: u32) -> vec2<f32> {
    return vec2<f32>(half_to_f32(packed & 0xffffu), half_to_f32(packed >> 16u));
}

fn color_matrix(
    color: vec4<f32>,
    texel0: vec4<u32>,
    texel1: vec4<u32>,
    texel2: vec4<u32>,
) -> vec4<f32> {
    let m01 = unpack_halves(texel0.y);
    let m23 = unpack_halves(texel0.z);
    let m45 = unpack_halves(texel0.w);
    let m67 = unpack_halves(texel1.x);
    let m89 = unpack_halves(texel1.y);
    let m1011 = unpack_halves(texel1.z);
    let m1213 = unpack_halves(texel1.w);
    let m1415 = unpack_halves(texel2.x);
    let m1617 = unpack_halves(texel2.y);
    let m1819 = unpack_halves(texel2.z);

    var c = vec4<f32>(0.0);
    if color.a > 0.0 {
        c = vec4<f32>(color.rgb / color.a, color.a);
    }
    let r = m01.x * c.r + m01.y * c.g + m23.x * c.b + m23.y * c.a + m45.x;
    let g = m45.y * c.r + m67.x * c.g + m67.y * c.b + m89.x * c.a + m89.y;
    let b = m1011.x * c.r + m1011.y * c.g + m1213.x * c.b + m1213.y * c.a + m1415.x;
    let a = m1415.y * c.r + m1617.x * c.g + m1617.y * c.b + m1819.x * c.a + m1819.y;
    let out = clamp(vec4<f32>(r, g, b, a), vec4<f32>(0.0), vec4<f32>(1.0));
    return vec4<f32>(out.rgb * out.a, out.a);
}
