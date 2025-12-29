use math_tool::matrix::Matrix;


#[test]
fn lu() {
    let mut mat = Matrix::random_matrix(5, 3, -100.0, 100.0, false).round(0);
    let (l, u, p) = mat.lu();
    assert_eq!((&(&l * &u) - &(&p * &mat)).round(8).entries, Matrix::zeros(5, 3).entries);

    let mut mat = Matrix::random_matrix(4, 10, -100.0, 100.0, true).round(0);
    let (l, u, p) = mat.lu();
    assert_eq!((&(&l * &u) - &(&p * &mat)).round(8).entries, Matrix::zeros(4, 10).entries);
}

#[test]
fn svd() {
    let mut mat = Matrix::random_matrix(5, 3, -100.0, 100.0, false).round(0);
    let (u, s, vt) = mat.svd().unwrap();
    assert_eq!((&(&(&u * &s) * &vt) - &mat).round(8).entries, Matrix::zeros(5, 3).entries);

    let mut mat = Matrix::random_matrix(3, 5, -100.0, 100.0, false).round(0);
    let (u, s, vt) = mat.svd().unwrap();
    assert_eq!((&(&(&u * &s) * &vt) - &mat).round(8).entries, Matrix::zeros(3, 5).entries);

    let mut mat = Matrix::random_matrix(5, 5, -100.0, 100.0, false).round(0);
    let t = mat.svd().unwrap();
    assert_eq!((&(&(&t.0 * &t.1) * &t.2) - &mat).round(8).entries, Matrix::zeros(5, 5).entries);

}