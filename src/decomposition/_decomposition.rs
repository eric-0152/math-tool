use crate::matrix::Matrix;
use crate::vector::Vector;

impl Matrix {
    /// Return a tuple (Q, R).
    /// 
    /// <br>
    /// 
    /// ### *QR* Decomposition:
    /// &emsp; ***A*** = ***QR***, ***Q*** is a matrix contains orthonormal basis
    /// , from doing ***Gram Schmidt*** process. ***R*** is a upper triangular 
    /// matrix contains inner products with ***A***.
    pub fn qr_decomposition(self: &Self) -> Result<(Matrix, Matrix), String> {
        match self.gram_schmidt() {
            Ok(matrix_q) => {
                let mut matrix_r: Matrix = Matrix::zeros(self.col, self.col);
                for r in 0..matrix_r.row {
                    let orthonormal_col: Vector = matrix_q.get_column_vector(r).unwrap();
                    for c in r..matrix_r.col {
                        matrix_r.entries[r][c] = self
                            .get_column_vector(c)
                            .unwrap()
                            .inner_product(&orthonormal_col)
                            .unwrap();
                    }
                }

                Ok((matrix_q, matrix_r))
            }
    
            Err(error_msg) => {
                Err(error_msg)
            }

        }
    }

    /// Return a tuple (L, U, P).
    ///  
    /// <br>
    /// 
    /// ### *LU* Decomposition: 
    /// &emsp; ***A*** = ***LU***, ***L*** is lower triagular, and ***U*** is upper triangular.
    /// 
    /// <br>
    /// 
    /// The algorithm will swap rows if needed (diagnal has 0), if the order of rows is 
    /// important, use swap_with_permutation() with P to yield the correct order.
    pub fn lu_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        if self.row != self.col {
            return Err("Value Error: This matrix is not a square matrix.".to_string());
        }
        
        let mut matrix_u: Matrix = self.copy();
        let mut matrix_l = Matrix::zeros(self.row, self.row);
        let mut permutation = Matrix::identity(self.row);
        for c in 0..self.col {
            for r in (c + 1)..self.row {
                // If the pivot is 0.0, swap to non zero.
                if matrix_u.entries[c][c] == 0.0 {
                    for r in (c + 1)..matrix_u.row {
                        if matrix_u.entries[r][c] != 0.0 {
                            matrix_u = matrix_u.swap_row(c, r).unwrap();
                            matrix_l = matrix_l.swap_row(c, r).unwrap();
                            permutation = permutation.swap_row(c, r).unwrap();
                        }
                    }
                }

                matrix_l.entries[r][c] = matrix_u.entries[r][c] / matrix_u.entries[c][c]; 
                for e in 0..self.col {
                    matrix_u.entries[r][e] -= matrix_l.entries[r][c] * matrix_u.entries[c][e];
                }  
            }
        }    
        matrix_l = matrix_l.add_Matrix(Matrix::identity(self.row)).unwrap();

        Ok((matrix_l, matrix_u, permutation))
    }

    

    /// Return a tuple (L, D, V, P). 
    /// 
    /// <br>
    /// 
    /// ### *LDV* Decomposition:
    /// &emsp; ***A*** = ***LDV***, ***L*** is lower triagular, ***D*** is diagonal, and ***V*** 
    /// is upper triangular.  
    /// 
    /// <br>
    /// 
    /// The algorithm will swap rows if needed (diagnal has 0), if the order of rows is     
    /// important, use swap_with_permutation() with P to yield the correct order.
    pub fn ldv_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix, Matrix), String> {
        match self.lu_decomposition() {
            Ok(tuple) => {
                let matrix_l: Matrix = tuple.0;
                let matrix_u: Matrix = tuple.1;
                let permutation: Matrix = tuple.2;
                let mut matrix_d: Matrix = Matrix::zeros(matrix_u.row, matrix_u.col);
                for d in 0..matrix_d.row {
                    matrix_d.entries[d][d] = matrix_u.entries[d][d];
                }

                match matrix_d.inverse() {
                    Ok(inverse_v) => {
                        let matrix_v: Matrix = inverse_v.multiply_Matrix(&matrix_u).unwrap();
                        Ok((matrix_l, matrix_d, matrix_v, permutation))
                    }

                    Err(err_msg) => {
                        Err(err_msg)
                    }
                }
            }

            Err(err_msg) => {
                Err(err_msg)
            }
        }
    
    }

    ///  Return a tuple (L, L_T).
    /// 
    /// ### Cholesky Decomposition:
    /// &nbsp; 
    pub fn cholesky_decomposition(self: &Self) -> Result<(Matrix, Matrix), String> {
        match self.is_positive_definite() {
            x if x == true => {
                let mut matrix_l: Matrix = Matrix::zeros(self.row, self.col);
                for r in 0..matrix_l.row {
                    for c in 0..(r + 1) {
                        let mut summation: f64 = 0.0;
                        if r == c {
                            for e in 0..c {
                                summation += matrix_l.entries[c][e].powi(2);
                            }
                            matrix_l.entries[r][c] = (self.entries[c][c] - summation).sqrt();
                        } else {
                            for e in 0..c {
                                summation += matrix_l.entries[r][e] * matrix_l.entries[c][e];
                            }
                            matrix_l.entries[r][c] = (self.entries[r][c] - summation) / matrix_l.entries[c][c];
                        }
                    }
                }

                return Ok((matrix_l.copy(), matrix_l.transpose()));
            }

            _ => {return Err("Value Error: This matrix is not a positive definite matrix.".to_string());}
        }

    }
}