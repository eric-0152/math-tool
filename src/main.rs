#![allow(dead_code, unused)]
#![allow(warnings)]


mod test;
use math_tool::{decomposition, eigen, io, matrix::Matrix, polynomial::Polynomial, multipoly::MultiPoly, process, regression, solve, transform, vector::Vector, *};
use num_complex::Complex64;
use rand_distr::num_traits::ConstOne;
fn main() {    
    let mut mat = Matrix::random_matrix(3, 3, -100.0, 100.0, true).round(0);
    let mut b: Vector = Vector::random_vector(4, -9.0, 9.0, false).round(0);

}

