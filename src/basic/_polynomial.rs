use crate::vector::Vector;
use num_complex::Complex64;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Coefficient of the poly is start from constant
#[derive(Clone, Debug)]
pub struct Polynomial {
    degree: usize,
    pub coeff: Vec<Complex64>,
}

impl Polynomial {
    pub fn new(coefficients: &Vec<Complex64>) -> Polynomial {
        if coefficients.len() == 0 {
            return Polynomial::new(&vec![Complex64::ZERO]);
        }

        let degree: usize = coefficients.len() - 1;
        Polynomial {
            degree,
            coeff: coefficients.clone(),
        }
        .remove_redundant()
    }
    /// Return a poly with m 0.
    pub fn zero() -> Polynomial {
        Polynomial {
            degree: 0,
            coeff: vec![Complex64::ZERO],
        }
    }
    /// Return a poly with m 1.
    pub fn one() -> Polynomial {
        Polynomial {
            degree: 0,
            coeff: vec![Complex64::ONE],
        }
    }
    
    pub fn degree(&self) -> usize {
        self.degree
    }
    
    pub fn display(self: &Self) {
        let mut show_im = false;
        for e in 0..self.coeff.len() {
            if self.coeff[e].im != 0.0 {
                show_im = true;
                break;
            }
        }

        print!("[");
        if show_im {
            for e in 0..self.coeff.len() {
                if self.coeff[e].re != 0.0 || self.coeff[e].im != 0.0 {
                    if self.coeff[e].im >= 0.0 {
                        print!(
                            "{}",
                            format!(
                                "({:>8?} {:>8}j)",
                                self.coeff[e].re,
                                format!("+ {:<?}", self.coeff[e].im.abs())
                            )
                        );
                    } else {
                        print!(
                            "{}",
                            format!(
                                "({:>8?} {:>8}j)",
                                self.coeff[e].re,
                                format!("- {:<?}", self.coeff[e].im.abs())
                            )
                        );
                    }
                    if e != 0 {
                        print!("x^{}", e);
                    }
                    if e != (self.coeff.len() - 1) {
                        print!(" + ");
                    }
                }
            }
        } else {
            for e in 0..self.coeff.len() {
                if self.coeff[e].re != 0.0 {
                    print!("{}", format!("({:>8?})", self.coeff[e].re));
                    if e != 0 {
                        print!("x^{}", e);
                    }
                    if e != (self.coeff.len() - 1) {
                        print!(" + ")
                    }
                }
            }
        }
        println!("], degree: {}", self.degree);
    }

    #[inline]
    pub fn to_same_size(
        polynomial1: &Polynomial,
        polynomial2: &Polynomial,
    ) -> (Polynomial, Polynomial) {
        let p1 = polynomial1.remove_redundant();
        let p2 = polynomial2.remove_redundant();
        if p1.coeff.len() < p2.coeff.len() {
            let mut result_poly1: Polynomial = p1.clone();
            while result_poly1.coeff.len() < p2.coeff.len() {
                result_poly1.coeff.push(Complex64::ZERO);
            }

            return (result_poly1, p2.clone());
        } else if p1.coeff.len() > p2.coeff.len() {
            let mut result_poly2: Polynomial = p2.clone();
            while result_poly2.coeff.len() < p1.coeff.len() {
                result_poly2.coeff.push(Complex64::ZERO);
            }

            return (polynomial1.clone(), result_poly2);
        }

        (p1, p2)
    }
    #[inline]
    pub fn round(self: &Self, digit: usize) -> Polynomial {
        let mut result_poly: Polynomial = 10.0_f64.powi(digit as i32) * self;
        for coeff in &mut result_poly.coeff {
            coeff.re = coeff.re.round();
            coeff.im = coeff.im.round();
        }

        &result_poly / 10.0_f64.powi(digit as i32)
    }
    #[inline]
    pub fn remove_redundant(self: &Self) -> Polynomial {
        let mut result_poly: Polynomial = self.clone();
        let mut row_index: usize = result_poly.coeff.len() - 1;
        while row_index > 0 && result_poly.coeff[row_index] == Complex64::ZERO {
            result_poly.coeff.pop();
            row_index -= 1;
        }

        result_poly.degree = result_poly.coeff.len() - 1;
        result_poly
    }

    #[inline]
    /// Return a tuple (quotient, remainder)
    pub fn divide_by(self: &Self, divider: &Polynomial) -> (Polynomial, Polynomial) {
        let divider = divider.remove_redundant();
        let mut remainder: Polynomial = self.remove_redundant();
        let mut quotient: Polynomial = Polynomial::zero();
        let degree_one = Polynomial::new(&vec![Complex64::ZERO, Complex64::ONE]);
        let mut degree_diff: usize = remainder.degree - divider.degree;
        loop {
            if remainder.coeff.len() < divider.coeff.len() {
                break;
            }

            let coefficient = remainder.coeff[remainder.degree] / divider.coeff[divider.degree];
            for d in 0..(divider.degree + 1) {
                remainder.coeff[d + degree_diff];
                remainder.coeff[d + degree_diff] -= coefficient * divider.coeff[d];
            }
            remainder = remainder.remove_redundant();
            quotient = &(&quotient * &degree_one) + &Polynomial::new(&vec![coefficient]);
            quotient.degree += 1;
            if degree_diff == 0 {
                break;
            } else {
                degree_diff -= 1;
            }
        }

        quotient.degree -= 1;
        (quotient, remainder)
    }

    #[inline]
    pub fn evaluate(self: &Self, value: &Complex64) -> Complex64 {
        let mut result: Complex64 = self.coeff[0];
        let mut power: i32 = 1;
        for d in 1..self.coeff.len() {
            result += self.coeff[d] * value.powi(power);
            power += 1;
        }

        result
    }

    #[inline]
    pub fn derivative(self: &Self) -> Polynomial {
        let mut result_poly: Polynomial = self.clone();
        for e in 1..self.coeff.len() {
            result_poly.coeff[e] = e as f64 * self.coeff[e];
        }
        if self.degree > 0 {
            result_poly.coeff.remove(0);
            result_poly.degree -= 1;
        }

        result_poly
    }
}

#[inline]
fn newton_raphson(poly: &Polynomial) -> Result<Complex64, String> {
    let derivative = poly.derivative();
    let mut x: Complex64 = Vector::random_vector(1, -1000.0, 1000.0, true).entries[0];
    if x.im == 0.0 {
        x.im = 1.0
    }
    const THRESHOLD: f64 = 1e-16;
    const MAX_ITER: u32 = 1000;
    let mut error: f64 = 1.0;
    let mut last_error: f64 = 0.0;
    let mut iter: u32 = 0;
    let mut old_x = x.clone();
    while (error - last_error).abs() > THRESHOLD && iter < MAX_ITER {
        let fx: Complex64 = poly.evaluate(&x);
        let dfx: Complex64 = derivative.evaluate(&x);
        x -= fx / dfx;
        if x.is_nan() || x.is_infinite() {
            if fx.re < 0.0 || dfx.re < 0.0 {
                if fx.re < 0.0 && dfx.re < 0.0 {
                    x = old_x + Complex64::new(10.0, 10.0);
                } else {
                    x = old_x - Complex64::new(10.0, 10.0);
                }
            } else {
                x = old_x + Complex64::new(10.0, 10.0);
            }
        }

        last_error = error;
        error = (&x - &old_x).norm();
        old_x = x;
        iter += 1;
    }

    if (error - last_error).abs() > 1e-8 {
        return Err("Failed to converge in Newton-Raphson method".to_string());
    }

    Ok(x)
}
fn _replace_value(value: Complex64) -> Complex64 {
    if value.is_nan() {
        Complex64::new(0.0, 0.0)
    } else if value.is_infinite() {
        Complex64::new(1e32, 1e32)
    } else {
        value
    }
}

#[inline]
pub fn laguerre_method(poly: &Polynomial) -> Result<Complex64, String> {
    let derivative: Polynomial = poly.derivative();
    let second_derivative: Polynomial = derivative.derivative();
    let mut x: Complex64 = Complex64::ZERO;
    let deg: f64 = poly.degree as f64;
    let deg_m: f64 = deg - 1.0;
    const THRESHOLD: f64 = 1e-16;
    const MAX_ITER: u32 = 1000;
    let mut iter: u32 = 0;
    while poly.evaluate(&x).norm() > THRESHOLD && iter < MAX_ITER {
        let fx: Complex64 = poly.evaluate(&x);
        let dfx: Complex64 = derivative.evaluate(&x);
        let ddfx: Complex64 = second_derivative.evaluate(&x);
        let g: Complex64 = dfx / fx;
        let h: Complex64 = g.powi(2) - (ddfx / fx);
        let add: Complex64 = g + (deg_m * ((deg * h) - g.powi(2))).sqrt();
        let sub: Complex64 = g - (deg_m * ((deg * h) - g.powi(2))).sqrt();
        if add.norm_sqr() >= sub.norm_sqr() {
            x -= deg / add;
        } else {
            x -= deg / sub;
        }

        iter += 1;
    }
    println!("{iter}");
    Ok(x)
}

/// Return a complex root using Newton-Raphson method.
pub fn find_root(poly: &Polynomial) -> Result<Vector, String> {
    let mut roots = Vector::new(&vec![]);
    let mut current_poly = poly.clone();
    let mut last_residual = Complex64::ZERO;
    while current_poly.coeff.len() > 1 {
        match laguerre_method(&current_poly) {
            Err(error_msg) => {
                return Err(error_msg);
            }
            Ok(new_root) => {
                
                roots = roots.append(&Vector::new(&vec![new_root]));
                let (quotient, remainder) =
                    current_poly.divide_by(&Polynomial::new(&vec![-new_root, Complex64::ONE]));
                current_poly = quotient;
                last_residual = remainder.coeff[0];
            }
        }
    }
    let last_index = roots.size() - 1;
    roots.entries[last_index] -= last_residual;
    roots
        .entries
        .sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    roots.entries.reverse();
    Ok(roots)
}




impl Add<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, float: f64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        result_poly.coeff[0] += float;

        result_poly.remove_redundant()
    }
}
impl Add<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn add(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        result_poly.coeff[0] += self;

        result_poly.remove_redundant()
    }
}
impl Add<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, complex: Complex64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        result_poly.coeff[0] += complex;

        result_poly.remove_redundant()
    }
}
impl Add<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn add(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        result_poly.coeff[0] += self;

        result_poly.remove_redundant()
    }
}
impl Add<&Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, poly: &Polynomial) -> Self::Output {
        let (mut p1, p2): (Polynomial, Polynomial) = Polynomial::to_same_size(self, poly);
        for d in 0..p1.coeff.len() {
            p1.coeff[d] += p2.coeff[d]
        }

        p1.degree = p1.coeff.len() - 1;
        p1.remove_redundant()
    }
}

impl Sub<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, float: f64) -> Self::Output {
        self + (-float)
    }
}
impl Sub<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn sub(self, poly: &Polynomial) -> Self::Output {
        self + &(-1.0 * poly)
    }
}
impl Sub<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, complex: Complex64) -> Self::Output {
        self + (-complex)
    }
}
impl Sub<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn sub(self, poly: &Polynomial) -> Self::Output {
        self + &(-1.0 * poly)
    }
}
impl Sub<&Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, poly: &Polynomial) -> Self::Output {
        self + &(-1.0 * poly)
    }
}

impl Mul<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= scalar;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn mul(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= self;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, scalar: Complex64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= scalar;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn mul(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= self;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<&Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, poly: &Polynomial) -> Self::Output {
        let p1: Polynomial = self.clone();
        let p2: Polynomial = poly.clone();

        if p1.degree == 0 {
            if p1.coeff.len() == 0 {
                return Polynomial::new(&vec![Complex64::ZERO]);
            } else {
                return p1.coeff[0] * &p2;
            }
        } else if p2.degree == 0 {
            if p2.coeff.len() == 0 {
                return Polynomial::new(&vec![Complex64::ZERO]);
            } else {
                return p2.coeff[0] * &p1;
            }
        }

        let mut result_poly: Polynomial = Polynomial::new(&vec![Complex64::ZERO]);
        while result_poly.degree < p1.degree + p2.degree {
            result_poly.coeff.push(Complex64::ZERO);
            result_poly.degree += 1;
        }
        for i in 0..p1.coeff.len() {
            for j in 0..p2.coeff.len() {
                result_poly.coeff[i + j] += p1.coeff[i] * p2.coeff[j];
            }
        }

        result_poly.remove_redundant()
    }
}
impl Div<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn div(self, poly: &Polynomial) -> Self::Output {
        poly * (1.0 / self)
    }
}
impl Div<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn div(self, poly: &Polynomial) -> Self::Output {
        poly * (1.0 / self)
    }
}
impl Div<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn div(self, float: f64) -> Self::Output {
        self * (1.0 / float)
    }
}
impl Div<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn div(self, complex: Complex64) -> Self::Output {
        self * (1.0 / complex)
    }
}

impl AddAssign for Polynomial {
    #[inline]
    fn add_assign(&mut self, other: Polynomial) {
        *self = &self.clone() + &other;
    }
}
impl AddAssign<f64> for Polynomial {
    #[inline]
    fn add_assign(&mut self, other: f64) {
        *self = &self.clone() + other;
    }
}
impl AddAssign<Complex64> for Polynomial {
    #[inline]
    fn add_assign(&mut self, other: Complex64) {
        *self = &self.clone() + other;
    }
}
impl SubAssign for Polynomial {
    #[inline]
    fn sub_assign(&mut self, other: Polynomial) {
        *self = &self.clone() - &other;
    }
}
impl SubAssign<f64> for Polynomial {
    #[inline]
    fn sub_assign(&mut self, other: f64) {
        *self = &self.clone() - other;
    }
}
impl SubAssign<Complex64> for Polynomial {
    #[inline]
    fn sub_assign(&mut self, other: Complex64) {
        *self = &self.clone() - other;
    }
}
impl Neg for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn neg(self: Self) -> Polynomial {
        -1.0 * &self.clone()
    }
}
impl MulAssign for Polynomial {
    #[inline]
    fn mul_assign(&mut self, other: Polynomial) {
        *self = &self.clone() * &other;
    }
}
impl MulAssign<f64> for Polynomial {
    #[inline]
    fn mul_assign(&mut self, other: f64) {
        *self = &self.clone() * other;
    }
}
impl MulAssign<Complex64> for Polynomial {
    #[inline]
    fn mul_assign(&mut self, other: Complex64) {
        *self = &self.clone() * other;
    }
}
impl DivAssign<f64> for Polynomial {
    #[inline]
    fn div_assign(&mut self, other: f64) {
        *self = &self.clone() / other;
    }
}
impl DivAssign<Complex64> for Polynomial {
    #[inline]
    fn div_assign(&mut self, other: Complex64) {
        *self = &self.clone() / other;
    }
}
