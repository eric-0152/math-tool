use math_tool::{vector::Vector, regression};

#[test]
fn polynomial() {
    const ROUND: usize = 2;
    const DEGREE: usize = 5;
    const DATA_SIZE: usize = 1000;
    const DATA_RANGE: (f64, f64) = (-100.0, 100.0);
    const PARAM_RANGE: (f64, f64) = (-100.0, 100.0);
    let params = Vector::random_vector(DEGREE + 1, PARAM_RANGE.0, PARAM_RANGE.1, true).round(ROUND);
    let x = Vector::linspace(DATA_RANGE.0, DATA_RANGE.1, DATA_SIZE);
    let y = regression::polynomial_data(&x, &params);
    let y_hat = regression::polynomial_regression(&x, &y, DEGREE).unwrap().round(ROUND);
    assert_eq!(y_hat.entries, params.entries);
}

#[test]
fn exponential() {
    const ROUND: usize = 8;
    const DATA_SIZE: usize = 1000;
    const DATA_RANGE: (f64, f64) = (-100.0, 100.0);
    const PARAM_RANGE: (f64, f64) = (-5.0, 5.0);
    let mut params = Vector::random_vector(2, PARAM_RANGE.0, PARAM_RANGE.1, false).round(ROUND);
    let x = Vector::linspace(DATA_RANGE.0, DATA_RANGE.1, DATA_SIZE);
    let y = regression::exponential_data(&x, &params);
    let y_hat = regression::exponential_regression(&x, &y).unwrap().round(ROUND);
    assert_eq!(y_hat.entries, params.entries);
}

#[test]
fn gaussian() {
    const ROUND: usize = 8;
    const DATA_SIZE: usize = 1000;
    const DATA_RANGE: (f64, f64) = (-100.0, 100.0);
    const PARAM_RANGE: (f64, f64) = (5.0, 100.0);
    let mut params = Vector::random_vector(2, PARAM_RANGE.0, PARAM_RANGE.1, false).round(ROUND);
    let x = Vector::linspace(DATA_RANGE.0, DATA_RANGE.1, DATA_SIZE);
    let y = regression::gaussian_1d_data(&x, &params);
    let y_hat = regression::gaussian_1d_regression(&x, &y).unwrap().round(ROUND);
    assert_eq!(y_hat.entries, params.entries);
}
