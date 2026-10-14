// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The color matrix filter.

/// A color matrix filter (`feColorMatrix` with a 4x5 matrix): each output
/// channel is a linear combination of the input's unpremultiplied red, green,
/// blue and alpha, plus an offset, clamped to [0, 1]. The CSS filter functions
/// `grayscale()`, `sepia()`, `saturate()`, `hue-rotate()`, `invert()`,
/// `opacity()`, `brightness()` and `contrast()` are all color matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMatrix {
    /// Row-major: 4 rows (R, G, B, A) of 5 columns (R, G, B, A, offset).
    pub matrix: [f32; 20],
}

impl ColorMatrix {
    /// Create a new color matrix filter.
    pub fn new(matrix: [f32; 20]) -> Self {
        Self { matrix }
    }

    /// The 20 coefficients as IEEE 754 half floats, two per `u32` (the
    /// first of each pair in the low bits): compact enough for a GPU filter
    /// record, and precise enough for 8-bit color.
    pub fn packed_f16(&self) -> [u32; 10] {
        let mut out = [0_u32; 10];
        for (i, pair) in out.iter_mut().enumerate() {
            *pair = u32::from(f16_bits(self.matrix[2 * i])) | (u32::from(f16_bits(self.matrix[2 * i + 1])) << 16);
        }
        out
    }

    /// Apply the matrix to one premultiplied RGBA color; the result is
    /// premultiplied.
    pub fn apply_premultiplied(&self, rgba: [f32; 4]) -> [f32; 4] {
        let a = rgba[3];
        let unpremultiplied = if a > 0.0 {
            [rgba[0] / a, rgba[1] / a, rgba[2] / a, a]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        let m = &self.matrix;
        let row = |r: usize| {
            let c = &m[r * 5..r * 5 + 5];
            (c[0] * unpremultiplied[0]
                + c[1] * unpremultiplied[1]
                + c[2] * unpremultiplied[2]
                + c[3] * unpremultiplied[3]
                + c[4])
                .clamp(0.0, 1.0)
        };
        let out_a = row(3);
        [row(0) * out_a, row(1) * out_a, row(2) * out_a, out_a]
    }
}

/// IEEE 754 half-precision bits of `x`, rounded to nearest even.
#[expect(clippy::cast_possible_truncation, reason = "bit manipulation of f16 fields")]
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity or NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let round = |value: u32, remainder: u32, halfway: u32| {
        if remainder > halfway || (remainder == halfway && value & 1 == 1) { value + 1 } else { value }
    };
    if e <= 0 {
        // Subnormal (or zero) in f16.
        if e < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let value = full >> shift;
        return sign | round(value, full & ((1 << shift) - 1), 1 << (shift - 1)) as u16;
    }
    // A carry out of the mantissa correctly rounds up into the exponent.
    sign | round(((e as u32) << 10) | (mantissa >> 13), mantissa & 0x1fff, 0x1000) as u16
}

#[cfg(test)]
mod tests {
    use super::{ColorMatrix, f16_bits};
    use crate::filter_effects::matrices;

    #[test]
    fn identity_and_grayscale() {
        let c = [0.25, 0.5, 0.125, 0.5];
        assert_eq!(ColorMatrix::new(matrices::IDENTITY).apply_premultiplied(c), c);
        let g = ColorMatrix::new(matrices::GRAYSCALE).apply_premultiplied([1.0, 0.0, 0.0, 1.0]);
        assert!((g[0] - 0.2126).abs() < 1e-6 && (g[1] - 0.2126).abs() < 1e-6 && g[3] == 1.0);
    }

    #[test]
    fn transparent_stays_transparent_and_output_is_clamped() {
        let m = ColorMatrix::new([
            2.0, 0.0, 0.0, 0.0, 0.5, //
            0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        assert_eq!(m.apply_premultiplied([0.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.apply_premultiplied([0.8, 0.0, 0.0, 1.0])[0], 1.0);
    }

    fn decode(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = i32::from((h >> 10) & 0x1f);
        let m = f32::from(h & 0x3ff);
        sign * if e == 0 { m * 2f32.powi(-24) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
    }

    #[test]
    fn half_precision_bits() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(0.0), 0);
        for x in [0.2126_f32, 0.7152, 0.0722, 0.393, -0.2, 1.4, 0.1, 1e-5, 3.75] {
            let y = decode(f16_bits(x));
            assert!((y - x).abs() <= x.abs() * 1e-3 + 6e-8, "{x} -> {y}");
        }
    }
}
