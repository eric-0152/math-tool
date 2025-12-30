use math_tool::{matrix::Matrix, solve, *};

#[test]
fn null_space() {
    let matrix = to_matrix!([1, -1, -1], [2, -2, 1]);
    let answer = to_matrix!([1], [1], [0]);
    let output = solve::null_space(&matrix);
    assert_eq!(output.entries, answer.entries);

    let matrix = to_matrix!([2, 1, 6, 6], [2, 5, -8, -3]);
    let answer = to_matrix!([-4.125, -4.75], [2.25, 3.5], [0, 1], [1, 0]);
    let output = solve::null_space(&matrix);
    assert_eq!(output.round(5).entries, answer.entries);
}
