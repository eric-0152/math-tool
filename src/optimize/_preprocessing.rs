use crate::matrix::Matrix;
use crate::vector::Vector;
use num_complex::Complex64;

/// ### Normalize Formula :
/// &emsp; ***(x - min) / (max - min)***
///
/// ### Return the vector after normalization.
/// ### Only modify the real part of vector.
pub fn normalize(data: &Vector) -> Vector {
    if data.size == 0 {
        return data.clone();
    }

    let mut result_vector: Vector = Vector::zeros(data.size);
    let mut min: f64 = data.entries[0].re;
    let mut max: f64 = data.entries[0].re;
    for e in 0..data.size {
        if data.entries[e].re > max {
            max = data.entries[e].re
        } else if data.entries[e].re < min {
            min = data.entries[e].re
        }
    }

    for e in 0..data.size {
        result_vector.entries[e].re = (data.entries[e].re - min) / (max - min);
    }

    result_vector
}

/// ### Principle Component Analysis
///
/// ### Return the matrix after PCA.
pub fn pca(matrix: &Matrix, dimension: usize) -> Result<Matrix, String> {
    if dimension > matrix.shape.0 {
        return Err(
            "Input Error: Parameter dimension cannot be greater than matrix's row".to_string(),
        );
    }

    let mut row_mean: Matrix = Matrix::zeros(matrix.shape.0, 1);
    for r in 0..matrix.shape.0 {
        let mut sum: Complex64 = Complex64::ZERO;
        for c in 0..matrix.shape.1 {
            sum += matrix.entries[r][c];
        }
        row_mean.entries[r][0] = sum / matrix.shape.1 as f64;
    }
    let mean_matrix: Matrix = &row_mean * &Matrix::ones(1, matrix.shape.1);
    let residual_matrix: Matrix = matrix - &mean_matrix;
    let (u, _, _) = residual_matrix.svd()?;
    let mut principle_component: Matrix = Matrix::zeros(0, 0);
    for d in 0..dimension {
        if d == u.shape.1 {
            break;
        }
        principle_component = principle_component.append_vector(&u.get_column_vector(d)?, 0)?;
    }

    let result_matrix: Matrix = &principle_component * &mean_matrix;
    Ok(result_matrix)
}
