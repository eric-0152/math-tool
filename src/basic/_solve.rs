use crate::matrix::Matrix;
use crate::vector::Vector;

/// Given a upper triangular matrix ***A*** and vector ***b***, return a vector ***x***
/// such that ***Ax*** = ***b***.
pub fn upper_triangular(matrix: &Matrix, b: &Vector) -> Result<Vector, String> {
    if matrix.row != b.size {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if !matrix.is_upper_triangular() {
        return Err("Input Error: The input matrix is not upper triangular.".to_string());
    }

    let mut vector_x: Vector = Vector::zeros(matrix.col);
    let min_range: usize = matrix.row.min(matrix.col);
    for diag in (0..min_range).rev() {
        vector_x.entries[diag] = b.entries[diag] / matrix.entries[diag][diag];
        for prev in ((diag + 1)..min_range).rev() {
            vector_x.entries[diag] -=
                matrix.entries[diag][prev] * vector_x.entries[prev] / matrix.entries[diag][diag];
        }
    }

    // Check consistency
    for e in 0..vector_x.size {
        if vector_x.entries[e].is_nan() {
            return Err("Value Error: The system is not consistent".to_string());
        }
    }

    Ok(vector_x)
}

/// Given a lower triangular matrix ***A*** and vector ***b***, return a vector ***x***
/// such that ***Ax*** = ***b***.
pub fn lower_triangular(matrix: &Matrix, b: &Vector) -> Result<Vector, String> {
    if matrix.row != b.size {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if !matrix.is_lower_triangular() {
        return Err("Input Error: The input matrix is not lower triangular.".to_string());
    }

    let mut vector_x: Vector = Vector::zeros(matrix.col);
    let min_range = matrix.row.min(matrix.col);
    for diag in 0..min_range {
        vector_x.entries[diag] = b.entries[diag] / matrix.entries[diag][diag];
        for prev in 0..diag {
            vector_x.entries[diag] -=
                matrix.entries[diag][prev] * vector_x.entries[prev] / matrix.entries[diag][diag];
        }
    }

    // Check consistency
    for e in 0..vector_x.size {
        if vector_x.entries[e].is_nan() {
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
    b: &Vector,
) -> Result<(Matrix, Vector, Matrix), String> {
    if matrix.row != b.size {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    }

    // Reduve to upper triangular form.
    let mut result_matrix: Matrix = matrix.clone();
    let mut result_vector: Vector = b.clone();
    let mut permutation = Matrix::identity(matrix.row);
    for d in 0..result_matrix.col.min(result_matrix.row) {
        // If the pivot is 0.0, swap to non zero.
        if result_matrix.entries[d][d] == 0.0 {
            let mut is_swap = false;
            for r in (d + 1)..result_matrix.row {
                if result_matrix.entries[r][d] != 0.0 {
                    result_matrix = result_matrix.swap_row(d, r).unwrap();
                    result_vector = result_vector.swap_element(d, r).unwrap();
                    permutation = permutation.swap_row(d, r).unwrap();
                    is_swap = true;
                    break;
                }
            }
            if !is_swap {
                continue;
            }
        }

        for r in (d + 1)..result_matrix.row {
            let scale: f64 = result_matrix.entries[r][d] / result_matrix.entries[d][d];
            result_vector.entries[r] -= scale * result_vector.entries[d];
            for e in 0..matrix.col {
                result_matrix.entries[r][e] -= scale * result_matrix.entries[d][e];
            }
        }
    }

    // Reduce to diagonal form
    for c in (0..result_matrix.col.min(result_matrix.row)).rev() {
        if result_matrix.entries[c][c] == 0.0 {
            continue;
        }

        for r in (0..c.min(result_matrix.row)).rev() {
            let scale = result_matrix.entries[r][c] / result_matrix.entries[c][c];
            result_matrix.entries[r][c] -= scale * result_matrix.entries[c][c];
            result_vector.entries[r] -= scale * result_vector.entries[c];
        }
    }

    // Pivots -> 1
    for r in 0..result_matrix.row {
        for c in r..result_matrix.col {
            if result_matrix.entries[r][c] != 0.0 {
                let scale: f64 = result_matrix.entries[r][c];
                for e in c..result_matrix.col {
                    result_matrix.entries[r][e] /= scale;
                }
                result_vector.entries[r] /= scale;

                break;
            }
        }
    }

    Ok((result_matrix, result_vector, permutation))
}
