use crate::matrix::Matrix;
use crate::vector::Vector;
use num_complex::Complex64;

impl Matrix {
    /// ### Return a matrix which contains orthonormal basis.
    pub fn gram_schmidt(self: &Self) -> Result<Matrix, String> {
        if self.shape.0 == 0 {
            return Err("Value Error: This matrix has no column.".to_string());
        }

        let mut matrix: Matrix = self.clone();
        if matrix.shape.0 < matrix.shape.1 {
            matrix = matrix.transpose();
        }

        let mut current_col: Vector = matrix.get_column_vector(0)?;
        let mut orthogonal_matrix: Matrix = current_col.as_matrix();
        for c in 1..matrix.shape.1 {
            current_col = matrix.get_column_vector(c)?;
            let mut new_orthogonal: Vector = current_col.clone();
            for pre_c in 0..c {
                let previous_orthogonal: Vector = orthogonal_matrix.get_column_vector(pre_c)?;
                let norm: f64 = previous_orthogonal.norm();
                let coefficient: Complex64 =
                    previous_orthogonal.inner_product(&current_col)? / norm.powi(2);
                new_orthogonal = &new_orthogonal - &(&previous_orthogonal * coefficient);
            }

            orthogonal_matrix = orthogonal_matrix.append_vector(&new_orthogonal, 1)?;
        }

        let mut orthonormal_matrix: Matrix = Matrix::zeros(0, 0);
        for c in 0..orthogonal_matrix.shape.1 {
            let column: Vector = orthogonal_matrix.get_column_vector(c).unwrap();
            let norm: f64 = column.norm();
            orthonormal_matrix = orthonormal_matrix.append_vector(&(&column / norm), 1)?;
        }

        Ok(orthonormal_matrix)
    }
}
