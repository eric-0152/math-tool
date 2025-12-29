use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::polynomial::{Polynomial, find_root};
use num_complex::Complex64;
use crate::solve;

pub fn rayleigh_quotient(eigenvector: &Vector, matrix_a: &Matrix) -> Complex64 {
    let numerator: Complex64 = (&eigenvector.transpose() * &(matrix_a * eigenvector)).entries[0];
    let denominator: Complex64 = (&eigenvector.transpose() * eigenvector).entries[0];

    numerator / denominator
}


/// ### Generate a similar matrix with QR decomposition.
/// ### Return a similar matrix.
///
/// ### Algorithm :
/// &emsp; ***A = QR*** => ***RQ = S*** => ***S = Q^(-1)QRQ*** => ***S = Q^(-1)AQ***
#[inline]
pub fn qr_similar_matrix(matrix: &Matrix) -> Result<Matrix, String> {
    if matrix.shape.0 != matrix.shape.1 {
        return Err("Input Error: The matrix is not square.".to_string());
    }
    
    match matrix.qr() {
        Ok((q, r)) => {
            Ok(&q * &r)
        }
        Err(error_msg) => Err(error_msg),
    }
}

#[inline]
fn no_check_qr_similar_matrix(matrix: &Matrix) -> Matrix {
    let tuple = matrix.qr().unwrap();

    &tuple.1 * &tuple.0
}

pub fn shift_qr_algorithm(
    matrix: &Matrix,
    max_iter: u32,
    error_thershold: f64,
) -> Result<(Matrix, f64), String> {
    match qr_similar_matrix(matrix) {
        Ok(mut matrix_similar) => {
            let last_idx: usize = matrix_similar.shape.0 - 1;
            let matrix_size: usize = matrix_similar.shape.0;
            let mut last_eigenvalue: Complex64 = matrix_similar.entries[last_idx][last_idx];
            let mut error: f64 = 1.0;
            let mut step: u32 = 0;
            while error > error_thershold && step < max_iter {
                let shift: Matrix = &Matrix::identity(matrix_size) * last_eigenvalue;
                matrix_similar = &matrix_similar - &shift;
                matrix_similar = &no_check_qr_similar_matrix(&matrix_similar) + &shift;
                error = (matrix_similar.entries[last_idx][last_idx] - last_eigenvalue).norm();
                last_eigenvalue = matrix_similar.entries[last_idx][last_idx];
                step += 1;
            }
            Ok((matrix_similar, error))
        }

        Err(error_msg) => Err(error_msg),
    }
}


#[inline]
fn sub_determinant(poly_matrix: &Vec<Vec<Polynomial>>) -> Polynomial {
    if poly_matrix.len() == 2 {
        return &(&poly_matrix[0][0] * &poly_matrix[1][1]) - &(&poly_matrix[0][1] * &poly_matrix[1][0]);
    }

    let mut lambda_poly: Polynomial = Polynomial::zero();
    for r in 0..poly_matrix.len() {
        let mut sub_coefficient: Polynomial = poly_matrix[r][0].clone();
        if sub_coefficient.coeff == Polynomial::zero().coeff {
            continue;
        }
        
        if r % 2 == 1 {
            sub_coefficient = -1.0 * &sub_coefficient;
        }
        
        let mut sub_poly_matrix: Vec<Vec<Polynomial>> = poly_matrix.clone();
        sub_poly_matrix.remove(r);
        for c in 0..sub_poly_matrix.len() {
            sub_poly_matrix[c].remove(0);
        }

        lambda_poly = &lambda_poly + &(&sub_coefficient * &sub_determinant(&sub_poly_matrix));
    }

    lambda_poly
}

/// Return a vector contains polynomial coefficients(from constant to highest degree).
/// 
/// Using Faddeev-LeVerrier algorithm.
pub fn characteristic_polynomial(matrix: &Matrix) -> Result<Polynomial, String> {
    if matrix.shape.0 != matrix.shape.1 {
        return Err("Input Error: This matrix is not square.".to_string());
    }

    let mut m = Matrix::identity(matrix.shape.0);
    let mut c = Complex64::ONE;
    let mut coefficient = vec![c];
    for i in 0..matrix.shape.0 {
        let am = matrix * &m;
        c = -am.trace() / (i as f64 + 1.0);
        coefficient.push(c);
        m = am;
        for d in 0..m.shape.0 {
            m.entries[d][d] += c;
        }
    }

    coefficient.reverse();
    Ok(Polynomial::new(&coefficient))
}

/// Return a vector which contains eigenvalue of matrix and the difference.
/// 
/// Use the qr algorithm first to eliminate the lower triangular part of the matrix.
pub fn eigenvalue(
    matrix: &Matrix
) -> Result<Vector, String> {
    if matrix.shape.0 != matrix.shape.1 {
        return Err("Input Error: This matrix is not square.".to_string());
    }
    
    let lambda_function = characteristic_polynomial(matrix)?;
    let eigenvalues = find_root(&lambda_function);        
    Ok(eigenvalues)
}

pub fn eigenvector(
    matrix: &Matrix,
    eigen_value: Complex64,
) -> Result<Matrix, String> {
    if matrix.shape.0 != matrix.shape.1 {
        return Err("Input Error: The input matrix is not square.".to_string());
    }
    
    let eigen_kernel: Matrix = &(&Matrix::identity(matrix.shape.0) * eigen_value) - matrix;
    Ok(solve::null_space(&eigen_kernel))
}
