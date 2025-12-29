use math_tool::{matrix::Matrix, polynomial::Polynomial, vector::Vector, *};
use num_complex::Complex64;

#[test]
fn characteristic_polynomial() {
    let matrix = to_matrix!(
        [7, -2, 9, -7],
        [9, 2, -6, 1],
        [7, -4, -1, -6],
        [-2, 2, -6, -2]);
    let answer = to_polynomial!([3678, 412, -132, -6, 1]);
    assert_eq!(eigen::characteristic_polynomial(&matrix).unwrap().coeff, answer.coeff);

    let matrix = to_matrix!(
        [6, 8, -7],
        [-1, 9, -4],
        [-5, -4, 8]);
    let answer = to_polynomial!([-217, 131, -23, 1]);
    assert_eq!(eigen::characteristic_polynomial(&matrix).unwrap().coeff, answer.coeff);
}

#[test]
fn eigenvalue() {
    const THERESHOLD: f64 = 1e-8;
    let matrix = Matrix::random_matrix(5, 5, -100.0, 100.0, false);
    let eigenvalue = eigen::eigenvalue(&matrix).unwrap();
    for e in 0..eigenvalue.size {
        let eigen_kernel: Matrix = &(&Matrix::identity(matrix.shape.0) * eigenvalue.entries[e]) - &matrix;
        assert!(eigen_kernel.determinant().unwrap().norm_sqr() < THERESHOLD);
    }
}


