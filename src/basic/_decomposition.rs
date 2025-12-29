use num_complex::Complex64;
use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::eigen;

impl Matrix {
    /// ### *LU* Decomposition :
    /// &emsp; ***A = LU***.
    ///
    /// ### Return a tuple (***L, U, P***) :
    /// &emsp; ***L*** : Lower triangular matrix. 
    /// 
    /// &emsp; ***U*** : Upper triangular matrix.
    /// 
    /// &emsp; ***P*** : Permutation matrix.
    pub fn lu(self: &Self) -> (Matrix, Matrix, Matrix) {
        let mut u: Matrix = self.clone();
        let mut l: Matrix = Matrix::zeros(self.shape.0, self.shape.0);
        let mut permutation: Matrix = Matrix::identity(self.shape.0);
        for c in 0..self.shape.1.min(self.shape.0) {
            // If the pivot is 0.0, swap to non zero.
            if u.entries[c][c] == Complex64::ZERO {
                let mut is_swap = false;
                for r in (c + 1)..u.shape.0 {
                    if u.entries[r][c] != Complex64::ZERO {
                        u = u.swap_row(c, r).unwrap();
                        l = l.swap_row(c, r).unwrap();
                        permutation = permutation.swap_row(c, r).unwrap();
                        is_swap = true;
                        break;
                    }
                }
                if !is_swap {
                    continue;
                }
            }

            for r in (c + 1)..self.shape.0 {
                l.entries[r][c] = u.entries[r][c] / u.entries[c][c];
                for e in 0..self.shape.1 {
                    let row_element: Complex64 = u.entries[c][e];
                    u.entries[r][e] -= l.entries[r][c] * row_element;
                }
            }
        }
        l = &l + &Matrix::identity(self.shape.0);
        (l, u, permutation)
    }

    /// ### *LDV* Decomposition :
    /// &emsp; ***A = LDV***.  
    ///
    /// ### Return a tuple (***L, D, V, P***) :
    /// &emsp; ***L*** : Lower triangular matrix.
    ///  
    /// &emsp; ***D*** : Diagonal matrix.
    ///  
    /// &emsp; ***V*** : Upper triangular matrix.
    ///  
    /// &emsp; ***P*** : Permutation matrix.
    pub fn ldv(self: &Self) -> Result<(Matrix, Matrix, Matrix, Matrix), String> {
        let (l, u, permutation) = self.lu();
        let mut d: Matrix = Matrix::identity(self.shape.0);
        for r in 0..self.shape.0.min(self.shape.1) {
            d.entries[r][r] = u.entries[r][r];
        }

        match d.inverse() {
            Err(err_msg) => Err(err_msg),
            Ok(inverse_d) => {
                let v: Matrix = &inverse_d * &u;
                Ok((l, d, v, permutation))
            }
        }
    }

    /// ### Cholesky Decomposition :
    /// &emsp; ***A = LL^T***.
    /// 
    /// ### Return a tuple (***L, L^T***) :
    /// &emsp; ***L*** : Lower triangular matrix.
    /// 
    /// &emsp; ***L^T*** : Transpose of ***L***.
    pub fn cholesky_decomposition(self: &Self) -> Result<(Matrix, Matrix), String> {
        if !self.is_positive_definite() {
            return Err("Value Error: This matrix is not a positive definite matrix.".to_string());
        }

        let mut l: Matrix = Matrix::zeros(self.shape.0, self.shape.1);
        for r in 0..l.shape.0 {
            for c in 0..(r + 1) {
                let mut summation: Complex64 = Complex64::ZERO;
                if r == c {
                    for e in 0..c {
                        summation += l.entries[c][e].powi(2);
                    }
                    l.entries[r][c] = (self.entries[c][c] - summation).sqrt();
                } else {
                    for e in 0..c {
                        summation += l.entries[r][e] * l.entries[c][e];
                    }
                    l.entries[r][c] = (self.entries[r][c] - summation) / l.entries[c][c];
                }
            }
        }

        Ok((l.clone(), l.transpose()))
    }

    /// ### ***LDLT*** Decomposition :
    /// &emsp; ***A = CC^T*** (from Cholesky decomposition) = ***LD^(1/2) @ (LD^(1/2))^T***
    /// = ***L @ D^(1/2)^2 @ L^T = LDL^T***.
    /// 
    /// ### Return a tuple (***L, D, L^T***) :
    /// &emsp; ***L*** : Lower triangular matrix.
    /// 
    /// &emsp; ***D*** : Diagonal matrix.
    /// 
    /// &emsp; ***L^T*** : Transpose of the lower triangular matrix.
    pub fn ldlt(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        match self.cholesky_decomposition() {
            Err(error_msg) => Err(error_msg),
            Ok((matrix_c, _)) => {
                let mut matrix_d: Matrix = Self::identity(self.shape.0);
                let mut matrix_l: Matrix = matrix_c.clone();
                for d in 0..matrix_l.shape.1 {
                    matrix_d.entries[d][d] = matrix_c.entries[d][d].powi(2);
                    let inverse_sqrt_diagnol: Complex64 = matrix_d.entries[d][d].sqrt();
                    for r in d..matrix_l.shape.0 {
                        matrix_l.entries[r][d] = matrix_c.entries[r][d] / inverse_sqrt_diagnol;
                    }
                }

                Ok((matrix_l.clone(), matrix_d, matrix_l.transpose()))
            }
        }
    }

    /// ### *QR* Decomposition :
    /// &emsp; ***A = QR***, ***Q***.
    /// 
    /// ### Return a tuple (***Q, R***) :
    /// &emsp; ***Q*** : Orthornormal matrix.
    /// 
    /// &emsp; ***R*** : Upper triangular matrix.
    pub fn qr(self: &Self) -> Result<(Matrix, Matrix), String> {
        match self.gram_schmidt() {
            Err(error_msg) => Err(error_msg),
            Ok(mut matrix_q) => {
                let mut matrix: Matrix = self.clone();
                let mut has_transpose: bool = false;
                if matrix.shape.0 < matrix.shape.1 {
                    has_transpose = true;
                    matrix = matrix.transpose();
                } 

                let mut matrix_r: Matrix = Matrix::zeros(matrix.shape.1, matrix.shape.1);
                for r in 0..matrix_r.shape.0 {
                    let orthonormal_col: Vector = matrix_q.get_column_vector(r)?;
                    for c in r..matrix_r.shape.1 {
                        matrix_r.entries[r][c] = matrix.get_column_vector(c)?.inner_product(&orthonormal_col)?;
                    }
                }
                
                if has_transpose {
                    let tmp: Matrix = matrix_q.transpose();
                    matrix_q = matrix_r.transpose();
                    matrix_r = tmp;
                }

                Ok((matrix_q, matrix_r))
            }
        }
    }

    /// ### Singular Value Decomposition :
    /// &emsp; ***A = U, Σ, V^T***
    /// 
    /// ### Return a tuple ***(U, Σ, V^T)*** :
    /// &emsp; ***U*** : Orthornormal matrix.
    /// 
    /// &emsp; ***Σ*** : Diagonal matrix contains singular values.
    /// 
    /// &emsp; ***V^T*** : Orthornormal matrix.
    pub fn svd(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        let ata: Matrix = &self.transpose() * self;
        match eigen::eigenvalue(&ata) {
            Err(error_msg) => Err(error_msg),
            Ok(mut eigenvalue) => {
                // Get eigenvalue
                const THERESHOLD: f64 = 1e-8;
                for e in (0..eigenvalue.size).rev() {
                    if eigenvalue.entries[e].re.abs() < THERESHOLD && eigenvalue.entries[e].im.abs() < THERESHOLD {
                        eigenvalue = eigenvalue.remove_element(e)?;
                    }
                }

                // Build V^T
                let mut vt: Matrix = Matrix::zeros(0, 0);
                for e in 0..eigenvalue.size {
                    let eigenvector: Matrix = eigen::eigenvector(&ata, eigenvalue.entries[e])?;
                    for c in 0..eigenvector.shape.1 {
                        vt = vt.append_matrix(&eigenvector.get_column_vector(c)?.normalize().transpose(), 0)?;
                    }
                }

                // Build Σ
                let sigma: Matrix = eigenvalue.square_root().to_diagonal();

                // Build U
                let mut u: Matrix = Matrix::zeros(0, 0);
                for r in 0..vt.shape.0 {
                    u = u.append_vector(&(&(self * &vt.get_row_vector(r)?) / sigma.entries[r][r]), 1)?;
                }

                Ok((u, sigma, vt))
            }
        }
    }
}
