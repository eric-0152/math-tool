use math_tool::{matrix::Matrix, to_matrix, *};

#[test]
fn inverse() {
    let matrix = to_matrix!([4, 8], [1, 10]);
    assert_eq!(
        (&matrix * &matrix.inverse().unwrap()).round(8).entries,
        Matrix::identity(2).round(8).entries
    );
    assert_eq!(
        (&matrix.inverse().unwrap() * &matrix).round(8).entries,
        Matrix::identity(2).round(8).entries
    );

    let matrix = to_matrix!([2, 4, 7], [5, 8, 14], [1, 4, -5]);
    assert_eq!(
        (&matrix * &matrix.inverse().unwrap()).round(8).entries,
        Matrix::identity(3).round(8).entries
    );
    assert_eq!(
        (&matrix.inverse().unwrap() * &matrix).round(8).entries,
        Matrix::identity(3).round(8).entries
    );
    // assert_eq!(matrix.inverse().unwrap().round(8).entries, answer.round(8).entries);

    let matrix = to_matrix!([6, 8, -7, 5], [-1, 9, -4, 2], [-5, -4, 8, 4]);
    assert!(matrix.inverse().is_err());

    let matrix = to_matrix!([5, 9, 3], [2, 5, 0], [4, 10, 0]);
    assert!(matrix.inverse().is_err());
}
