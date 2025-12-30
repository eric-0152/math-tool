use crate::vector::Vector;
use num_complex::Complex64;
use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    vec,
};

/// The coeff stores (scalar, [degree of variables])
#[derive(Debug, Clone)]
pub struct MultiPoly {
    pub names: Vec<String>,
    pub coeff: Vec<(Complex64, Vec<f64>)>,
}

impl MultiPoly {
    pub fn zero(names: &Vec<String>) -> Self {
        MultiPoly {
            names: names.clone(),
            coeff: vec![(Complex64::ZERO, vec![0.0; names.len()])],
        }
    }

    pub fn display(self: &Self) {
        let mut show_im = false;
        for e in 0..self.coeff.len() {
            if self.coeff[e].0.im != 0.0 {
                show_im = true;
                break;
            }
        }

        print!("[");
        if show_im {
            for e in 0..self.coeff.len() {
                if self.coeff[e].0.re != 0.0 || self.coeff[e].0.im != 0.0 {
                    if self.coeff[e].0.im >= 0.0 {
                        print!(
                            "{}",
                            format!(
                                "({:>8?} {:>8} j)",
                                self.coeff[e].0.re,
                                format!("+ {:<?}", self.coeff[e].0.im.abs())
                            )
                        );
                    } else {
                        print!(
                            "{}",
                            format!(
                                "({:>8?} {:>8} j)",
                                self.coeff[e].0.re,
                                format!("- {:<?}", self.coeff[e].0.im.abs())
                            )
                        );
                    }

                    for name in 0..self.names.len() {
                        if self.coeff[e].1[name] != 0.0 {
                            print!(
                                "({})",
                                format!("{}^{}", self.names[name], self.coeff[e].1[name])
                            );
                        }
                    }
                    if e != (self.coeff.len() - 1) {
                        print!(" + ");
                    }
                }
            }
        } else {
            for e in 0..self.coeff.len() {
                if self.coeff[e].0.re != 0.0 {
                    print!("{}", format!("({:>8?})", self.coeff[e].0.re));
                    for name in 0..self.names.len() {
                        if self.coeff[e].1[name] != 0.0 {
                            print!(
                                "({})",
                                format!("{}^{}", self.names[name], self.coeff[e].1[name])
                            );
                        }
                    }
                    if e != (self.coeff.len() - 1) {
                        print!(" + ")
                    }
                }
            }
        }
        println!("]");
    }

    pub fn combination(n: u32, r: u32) -> u32 {
        if r > n {
            return 0;
        }

        let mut result = 1;
        for i in 0..r {
            result *= n - i;
            result /= i + 1;
        }

        result
    }
    pub fn new(names: Vec<String>, coeff: Vec<(Complex64, Vec<f64>)>) -> Result<MultiPoly, String> {
        for d in 0..coeff.len() {
            if names.len() != coeff[d].1.len() {
                return Err(
                    "Input Error: The number of names and coefficients do not match.".to_string(),
                );
            }
        }

        Ok(MultiPoly { names, coeff })
    }

    #[inline]
    pub fn evaluate(self: &Self, value: &Vector) -> Result<Complex64, String> {
        if self.names.len() != value.size {
            return Err(
                "Input Error: The number of names and value's size do not match.".to_string(),
            );
        }

        let mut result: Complex64 = Complex64::new(0.0, 0.0);
        for (scalar, power) in self.coeff.iter() {
            let mut element: Complex64 = *scalar;
            for p in 0..power.len() {
                element *= value.entries[p].powf(power[p]);
            }

            result += element;
        }

        Ok(result)
    }

    pub fn partial_derivative(self: &Self, name: String) -> Result<MultiPoly, String> {
        let mut name_exist: bool = false;
        for params_name in self.names.clone() {
            if params_name == name {
                name_exist = true;
                break;
            }
        }
        if !name_exist {
            return Err(
                "Input Error: The name of parameter does not exist in the MultiPoly.".to_string(),
            );
        }

        let mut result_poly: MultiPoly = MultiPoly::zero(&self.names);
        let name_index: usize = self.names.iter().position(|x| *x == name).unwrap();
        for idx in (0..self.coeff.len()).rev() {
            if self.coeff[idx].1[name_index] != 0.0 {
                let new_scalar: Complex64 = self.coeff[idx].0 * self.coeff[idx].1[name_index];
                let mut new_power: Vec<f64> = self.coeff[idx].1.clone();
                new_power[name_index] -= 1.0;
                result_poly.coeff.push((new_scalar, new_power));
            }
        }

        Ok(result_poly.remove_redundant())
    }

    #[inline]
    pub fn remove_redundant(self: &Self) -> MultiPoly {
        // Find same name pair
        let mut same_name_idx: Vec<(usize, usize)> = Vec::new();
        let mut names = self.names.clone();
        for second in (1..self.names.len()).rev() {
            for first in 0..second {
                if self.names[second] == self.names[first] {
                    same_name_idx.push((second, first));
                    names.remove(second);
                    break;
                }
            }
        }

        // Build result_poly
        let mut result_poly: MultiPoly = MultiPoly {
            names,
            coeff: Vec::new(),
        };
        for d in (0..self.coeff.len()).rev() {
            if self.coeff[d].0 != Complex64::ZERO {
                let mut degrees: Vec<f64> = self.coeff[d].1.clone();
                for (second, first) in &same_name_idx {
                    degrees[*first] += degrees[*second];
                    degrees.remove(*second);
                }
                result_poly.coeff.push((self.coeff[d].0, degrees));
            }
        }

        // Combine same powers
        for second in (0..result_poly.coeff.len()).rev() {
            for first in (0..second).rev() {
                if result_poly.coeff[first].1 == result_poly.coeff[second].1 {
                    let second_coeff = result_poly.coeff[second].0;
                    result_poly.coeff[first].0 += second_coeff;
                    result_poly.coeff.remove(second);
                    break;
                }
            }
        }

        result_poly
    }
}

impl Add<f64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn add(self, float: f64) -> Self::Output {
        let mut result_poly: MultiPoly = self.clone();
        for e in 0..result_poly.coeff.len() {
            if result_poly.coeff[e].1 == vec![0.0; result_poly.names.len()] {
                result_poly.coeff[e].0 += float;
            }
        }

        result_poly.remove_redundant()
    }
}

impl Add<&MultiPoly> for f64 {
    type Output = MultiPoly;
    #[inline]
    fn add(self, multi_poly: &MultiPoly) -> Self::Output {
        let mut result_poly: MultiPoly = multi_poly.clone();
        for e in 0..result_poly.coeff.len() {
            if result_poly.coeff[e].1 == vec![0.0; result_poly.names.len()] {
                result_poly.coeff[e].0 += self;
            }
        }

        result_poly.remove_redundant()
    }
}

impl Add<Complex64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn add(self, complex: Complex64) -> Self::Output {
        let mut result_poly: MultiPoly = self.clone();
        for e in 0..result_poly.coeff.len() {
            if result_poly.coeff[e].1 == vec![0.0; result_poly.names.len()] {
                result_poly.coeff[e].0 += complex;
            }
        }

        result_poly.remove_redundant()
    }
}
impl Add<&MultiPoly> for Complex64 {
    type Output = MultiPoly;
    #[inline]
    fn add(self, multi_poly: &MultiPoly) -> Self::Output {
        let mut result_poly: MultiPoly = multi_poly.clone();
        for e in 0..result_poly.coeff.len() {
            if result_poly.coeff[e].1 == vec![0.0; result_poly.names.len()] {
                result_poly.coeff[e].0 += self;
            }
        }

        result_poly.remove_redundant()
    }
}
impl Add<&MultiPoly> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn add(self, multipoly: &MultiPoly) -> Self::Output {
        let mut result_poly = self.clone();
        result_poly.coeff.extend(multipoly.coeff.iter().cloned());
        result_poly.remove_redundant()
    }
}

impl Sub<f64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn sub(self, float: f64) -> Self::Output {
        self + -float
    }
}
impl Sub<&MultiPoly> for f64 {
    type Output = MultiPoly;
    #[inline]
    fn sub(self, multi_poly: &MultiPoly) -> Self::Output {
        self + &(-1.0 * multi_poly)
    }
}
impl Sub<Complex64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn sub(self, complex: Complex64) -> Self::Output {
        self + -complex
    }
}
impl Sub<&MultiPoly> for Complex64 {
    type Output = MultiPoly;
    #[inline]
    fn sub(self, multi_poly: &MultiPoly) -> Self::Output {
        self + &(-1.0 * multi_poly)
    }
}
impl Sub<&MultiPoly> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn sub(self, multipoly: &MultiPoly) -> Self::Output {
        self + &(-1.0 * multipoly)
    }
}
impl Mul<f64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        let mut result_poly = self.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d].0 *= scalar;
        }
        result_poly
    }
}
impl Mul<Complex64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn mul(self, scalar: Complex64) -> Self::Output {
        let mut result_poly = self.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d].0 *= scalar;
        }
        result_poly
    }
}
impl Mul<&MultiPoly> for f64 {
    type Output = MultiPoly;
    #[inline]
    fn mul(self, multipoly: &MultiPoly) -> Self::Output {
        let mut result_poly = multipoly.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d].0 *= self;
        }
        result_poly
    }
}

impl Mul<&MultiPoly> for Complex64 {
    type Output = MultiPoly;
    #[inline]
    fn mul(self, multipoly: &MultiPoly) -> Self::Output {
        let mut result_poly = multipoly.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d].0 *= self;
        }
        result_poly
    }
}

impl Mul for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn mul(self, multipoly: &MultiPoly) -> Self::Output {
        let mut names: Vec<String> = self.names.clone();
        names.append(&mut multipoly.names.clone());
        let mut result_poly: MultiPoly = MultiPoly {
            names,
            coeff: Vec::new(),
        };
        for i in 0..self.coeff.len() {
            for j in 0..multipoly.coeff.len() {
                let new_scalar: Complex64 = self.coeff[i].0 * multipoly.coeff[j].0;
                let mut new_power: Vec<f64> = self.coeff[i].1.clone();
                new_power.append(&mut multipoly.coeff[j].1.clone());
                result_poly.coeff.push((new_scalar, new_power));
            }
        }

        result_poly.remove_redundant()
    }
}
impl Div<&MultiPoly> for f64 {
    type Output = MultiPoly;
    #[inline]
    fn div(self, poly: &MultiPoly) -> Self::Output {
        poly * (1.0 / self)
    }
}
impl Div<&MultiPoly> for Complex64 {
    type Output = MultiPoly;
    #[inline]
    fn div(self, poly: &MultiPoly) -> Self::Output {
        poly * (1.0 / self)
    }
}

impl Div<f64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn div(self, float: f64) -> Self::Output {
        self * (1.0 / float)
    }
}
impl Div<Complex64> for &MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn div(self, complex: Complex64) -> Self::Output {
        self * (1.0 / complex)
    }
}

impl AddAssign for MultiPoly {
    #[inline]
    fn add_assign(&mut self, other: MultiPoly) {
        *self = &self.clone() + &other;
    }
}
impl AddAssign<f64> for MultiPoly {
    #[inline]
    fn add_assign(&mut self, other: f64) {
        *self = &self.clone() + other;
    }
}
impl AddAssign<Complex64> for MultiPoly {
    #[inline]
    fn add_assign(&mut self, other: Complex64) {
        *self = &self.clone() + other;
    }
}
impl SubAssign for MultiPoly {
    #[inline]
    fn sub_assign(&mut self, other: MultiPoly) {
        *self = &self.clone() - &other;
    }
}
impl SubAssign<f64> for MultiPoly {
    #[inline]
    fn sub_assign(&mut self, other: f64) {
        *self = &self.clone() - other;
    }
}
impl SubAssign<Complex64> for MultiPoly {
    #[inline]
    fn sub_assign(&mut self, other: Complex64) {
        *self = &self.clone() - other;
    }
}
impl Neg for MultiPoly {
    type Output = MultiPoly;
    #[inline]
    fn neg(self: Self) -> MultiPoly {
        -1.0 * &self.clone()
    }
}
impl MulAssign for MultiPoly {
    #[inline]
    fn mul_assign(&mut self, other: MultiPoly) {
        *self = &self.clone() * &other;
    }
}
impl MulAssign<f64> for MultiPoly {
    #[inline]
    fn mul_assign(&mut self, other: f64) {
        *self = &self.clone() * other;
    }
}
impl MulAssign<Complex64> for MultiPoly {
    #[inline]
    fn mul_assign(&mut self, other: Complex64) {
        *self = &self.clone() * other;
    }
}
impl DivAssign<f64> for MultiPoly {
    #[inline]
    fn div_assign(&mut self, other: f64) {
        *self = &self.clone() / other;
    }
}
impl DivAssign<Complex64> for MultiPoly {
    #[inline]
    fn div_assign(&mut self, other: Complex64) {
        *self = &self.clone() / other;
    }
}
