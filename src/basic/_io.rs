use num_complex::Complex64;
use crate::matrix::Matrix;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;

impl Matrix {
    /// The add or sub operations should be with no whitespace near.
    /// 
    /// ### Example
    /// 2 10 5+2i <br>
    /// 2-1i 49 3i
    pub fn read_txt(path: &str) -> Result<Matrix, String> {
        let openfile: Result<File, std::io::Error> = File::open(path);
        
        match openfile {
            Err(erroe_msg) => Err(erroe_msg.to_string()),
            Ok(file) => {
                let reader: BufReader<File> = BufReader::new(file);
                let mut rows: Vec<Vec<Complex64>> = Vec::new();
                for line in reader.lines() {
                    let line: String = line.unwrap();
                    let mut elements: std::str::SplitWhitespace<'_> = line.split_whitespace().into_iter();
                    let mut row: Vec<Complex64> = Vec::new();
                    loop {
                        let e: Option<&str> = elements.next();
                        if e.is_none() {
                            break;
                        } else {
                            match Complex64::from_str(e.unwrap()) {
                                Err(error_msg) => {
                                    return Err(error_msg.to_string());
                                }
                                Ok(num) => {
                                    row.push(num);
                                }
                            }
                        }
                    }
                    rows.push(row);
                }

                for r in 1..rows.len() {
                    if rows[0].len() != rows[r].len() {
                        return Err("Value Error: The size of rows are not match.".to_string());
                    }
                }

                Ok(Matrix::from_vec(&rows)?)
            }
        }
    }

    pub fn write_txt(matrix: &Matrix, path: &str, write_im: bool) -> Result<File, String> {
        let openfile: Result<File, std::io::Error> = File::create(path);
        match openfile {
            Err(error_msg) => Err(format!("Operation Error: {error_msg}.")),
            Ok(mut file) => {
                if write_im {
                    for r in 0..matrix.shape.0 {
                        for c in 0..matrix.shape.1 {
                            let re: f64 = matrix.entries[r][c].re; 
                            let im: f64 = matrix.entries[r][c].im; 
                            if im >= 0.0 {
                                write!(file, "{}+{}i ", re, im).expect("Write entrie.");
                            } else {
                                write!(file, "{}-{}i ", re, im).expect("Write entrie.");
                            }
                        }
                        write!(file, "\n").expect("Write new line.");
                    }
                } else {
                    for r in 0..matrix.shape.0 {
                        for c in 0..matrix.shape.1 {
                            write!(file, "{} ", matrix.entries[r][c].re).expect("Write entrie.");
                        }
                    }
                    write!(file, "\n").expect("Write new line.");
                }
                
                
                Ok(file)
            }
        }
    }
}
