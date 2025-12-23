use std::os::unix::raw::nlink_t;

use num_complex::Complex64;
use crate::matrix::Matrix;

/// Given a upper triangular matrix ***A*** and vector ***b***, return a vector ***x***
/// such that ***Ax*** = ***b***.
pub fn upper_triangular(matrix: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if matrix.shape.0 != b.shape.0 {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if !matrix.is_upper_triangular() {
        return Err("Input Error: The input matrix is not upper triangular.".to_string());
    }

    let mut vector_x: Matrix = Matrix::zeros(matrix.shape.1, 1);
    let min_range: usize = matrix.shape.0.min(matrix.shape.1);
    for diag in (0..min_range).rev() {
        vector_x.entries[diag][0] = b.entries[diag][0] / matrix.entries[diag][diag];
        for prev in ((diag + 1)..min_range).rev() {
            let element: Complex64 = vector_x.entries[prev][0];
            vector_x.entries[diag][0] -=
                matrix.entries[diag][prev] * element / matrix.entries[diag][diag];
        }
    }

    // Check consistency
    for e in 0..vector_x.shape.0 {
        if vector_x.entries[e][0].is_nan() {
            return Err("Value Error: The system is not consistent".to_string());
        }
    }

    Ok(vector_x)
}

/// Given a lower triangular matrix ***A*** and vector ***b***, return a vector ***x***
/// such that ***Ax*** = ***b***.
pub fn lower_triangular(matrix: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if matrix.shape.0 != b.shape.0 {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if !matrix.is_lower_triangular() {
        return Err("Input Error: The input matrix is not lower triangular.".to_string());
    } else if b.shape.1 != 1 {
        return Err("Input Error: The input b is not a vector.".to_string());
    }

    let mut vector_x: Matrix = Matrix::zeros(matrix.shape.1, 1);
    let min_range = matrix.shape.0.min(matrix.shape.1);
    for diag in 0..min_range {
        vector_x.entries[diag][0] = b.entries[diag][0] / matrix.entries[diag][diag];
        for prev in 0..diag {
            let element: Complex64 = vector_x.entries[prev][0];
            vector_x.entries[diag][0] -= matrix.entries[diag][prev] * element / matrix.entries[diag][diag];
        }
    }

    // Check consistency
    for e in 0..vector_x.shape.0 {
        if vector_x.entries[e][0].is_nan() {
            return Err("Value Error: The system is not consistent".to_string());
        }
    }

    Ok(vector_x)
}

/// Return the tuple contains matrix, b and permutation after Gaussian Jordan elimination.
///
/// The algorithm will swap rows if needed (diagnal has 0), if the order of rows is
/// important, use swap_with_permutation() to yield the correct order.
pub fn gauss_jordan_elimination(
    matrix: &Matrix,
    b: &Matrix,
) -> Result<(Matrix, Matrix, Matrix), String> {
    if matrix.shape.0 != b.shape.0 {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if b.shape.1 != 1 {
        return Err("Input Error: The input b is not a vector.".to_string());
    }

    // Reduce to upper triangular form.
    const THERESHOLD: f64 = 1e-8;
    let mut result_matrix: Matrix = matrix.clone();
    let mut result_vector: Matrix = b.clone();
    let mut permutation: Matrix = Matrix::identity(matrix.shape.0);
    let mut pivot_row: usize = 0;
    let mut pivot_col: usize = 0;
    let mut last_operate: i32 = 0;
    while pivot_row < result_matrix.shape.0 && pivot_col < result_matrix.shape.1 {
        // If the pivot is 0.0, swap to non zero.
        if result_matrix.entries[pivot_row][pivot_col].norm() < THERESHOLD {
            let mut is_swap = false;
            for r in (pivot_row + 1)..result_matrix.shape.0 {
                if result_matrix.entries[r][pivot_col] != Complex64::ZERO {
                    result_matrix = result_matrix.swap_row(pivot_row, r).unwrap();
                    result_vector = result_vector.swap_row(pivot_row, r).unwrap();
                    permutation = permutation.swap_row(pivot_row, r).unwrap();
                    is_swap = true;
                    break;
                }
            }
            if !is_swap {
                last_operate = 0;
                pivot_col += 1;
                continue;
            }
        }

        for r in (pivot_row + 1)..result_matrix.shape.0 {
            let scale: Complex64 = result_matrix.entries[r][pivot_col] / result_matrix.entries[pivot_row][pivot_col];
            let element: Complex64 = result_vector.entries[pivot_row][0];
            result_vector.entries[r][0] -= scale * element;
            for e in 0..matrix.shape.1 {
                let element: Complex64 = result_matrix.entries[pivot_row][e];
                result_matrix.entries[r][e] -= scale * element;
            }
        }
        pivot_row += 1;
        pivot_col += 1;
        last_operate = 1;
    }
    
    // Reduce to diagonal form
    if last_operate == 0 {
        pivot_col -= 1;
    } else if last_operate == 1 {
        pivot_row -= 1;
        pivot_col -= 1;
    }
    while pivot_row > 0 {
        for r in 0..pivot_row {
            if result_matrix.entries[pivot_row][pivot_col].norm() < THERESHOLD {
                continue;
            }
            let scale: Complex64 = result_matrix.entries[r][pivot_col] / result_matrix.entries[pivot_row][pivot_col];
            let element: Complex64 = result_vector.entries[pivot_row][0];
            result_vector.entries[r][0] -= scale * element;
            for e in pivot_col..result_matrix.shape.1 {
                let element: Complex64 = result_matrix.entries[pivot_row][e];
                result_matrix.entries[r][e] -= scale * element;
            }
        }
        pivot_row -= 1;
        pivot_col -= 1;
    }

    // Pivots -> 1
    for r in 0..result_matrix.shape.0 {
        for c in r..result_matrix.shape.1 {
            if result_matrix.entries[r][c] != Complex64::ZERO {
                let scale: Complex64 = result_matrix.entries[r][c];
                for e in c..result_matrix.shape.1 {
                    result_matrix.entries[r][e] /= scale;
                }
                result_vector.entries[r][0] /= scale;

                break;
            }
        }
    }

    Ok((result_matrix, result_vector, permutation))
}

pub fn null_space(matrix: &Matrix) -> Matrix {
    let rref: Matrix = gauss_jordan_elimination(matrix, &Matrix::zeros(matrix.shape.0, 1))
        .unwrap()
        .0;

    // Construct the matrix that contains relationship between each pivot and behind element.
    // Each column only contains two element.
    const THERESHOLD: f64 = 1e-8;
    let mut null_relate: Matrix = Matrix::zeros(0, 0);
    for r in (0..rref.shape.0.min(rref.shape.1)).rev() {
        let mut pivot = r;
        while rref.entries[r][pivot].norm() < THERESHOLD {
            pivot += 1;
            if pivot == rref.shape.1 {
                break;
            }
        }

        for right in (pivot + 1)..rref.shape.1 {
            if rref.entries[r][right].norm() < THERESHOLD {
                continue;
            }

            let mut relate_vector: Matrix = Matrix::zeros(rref.shape.1, 1);
            relate_vector.entries[pivot][0] = -1.0 * rref.entries[r][right];
            relate_vector.entries[right][0] = Complex64::ONE;
            null_relate = null_relate.append(&relate_vector, 1).unwrap();
        }
    }

    // Combine columns if has the same bottom value.
    let mut null_basis: Matrix = Matrix::zeros(0, 0);
    for r in (0..null_relate.shape.0).rev() {
        let mut null_vector: Matrix = Matrix::zeros(rref.shape.1, 1);
        null_vector.entries[r][0] = Complex64::ONE;
        for c in 0..null_relate.shape.1 {
            if null_relate.entries[r][c] == Complex64::ONE {
                for e in 0..r {
                    if null_relate.entries[e][c] != Complex64::ZERO {
                        null_vector.entries[e][0] = null_relate.entries[e][c];
                        break;
                    }
                }
            }
        }

        let mut element_num: i32 = 0;
        for e in 0..null_vector.shape.0 {
            if null_vector.entries[e][0] != Complex64::ZERO {
                element_num += 1;
            }
            if element_num == 2 {
                null_basis = null_basis.append(&null_vector, 0).unwrap();
                break;
            }
        }
    }

    // Complete the eigenvector
    for c in 0..rref.shape.1 {
        let mut zero_num: usize = 0;
        for r in 0..rref.shape.0 {
            if rref.entries[r][c] == Complex64::ZERO {
                zero_num += 1
            } else {
                break;
            }
        }

        if zero_num == rref.shape.0 {
            let mut zero_vector: Matrix = Matrix::zeros(rref.shape.1, 1);
            zero_vector.entries[c][0] = Complex64::ONE;
            null_basis = null_basis.append(&zero_vector, 1).unwrap();
        }
    }
    if null_basis.shape.0 == 0 {
        null_basis = null_basis
            .append(&Matrix::zeros(rref.shape.1, 1), 1)
            .unwrap();
    }

    null_basis
}
