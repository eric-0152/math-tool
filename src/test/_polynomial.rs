use math_tool::{*, polynomial, to_polynomial};

#[test]
fn divide_by() {
    let poly = to_polynomial!([1, 2, 3]);
    let divider = to_polynomial!([1, 1]);
    let (quotient, remainder) = poly.divide_by(&divider);
    assert_eq!(quotient.coeff, to_polynomial!([-1, 3]).coeff);
    assert_eq!(remainder.coeff, to_polynomial!([2]).coeff);
}
