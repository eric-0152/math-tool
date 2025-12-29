use crate::matrix::Matrix;
use crate::vector::Vector;

impl Matrix {
    /// ### Givens Rotation :
    /// &emsp; Rotate the matrix along the i-j plane, by counter-clockwise angle.
    /// 
    /// ### Return the matrix after rotation.
    pub fn givens_rotation(self: &Self, i: usize, j: usize, angle: f64) -> Result<Matrix, String> {
        if i >= self.shape.0 || j >= self.shape.0 {
            return Err("Input Error: Parameter i or j is out of bound.".to_string());
        }

        let mut rotation_matrix: Matrix = Matrix::identity(self.shape.0);
        rotation_matrix.entries[i][i].re = angle.cos();
        rotation_matrix.entries[j][j].re = angle.cos();
        rotation_matrix.entries[j][i].re = angle.sin();
        rotation_matrix.entries[i][j].re = -angle.sin();

        Ok(&rotation_matrix * self)
    }
    
    pub fn houserholder(vector: Vector) -> Matrix {
        &Matrix::identity(vector.size) - &(2.0 * &(&vector.as_matrix() * &vector.transpose()))
    }
}
