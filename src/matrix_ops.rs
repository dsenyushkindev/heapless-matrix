use heapless::Vec;

use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use core::cmp::PartialEq;

use core::usize;

use crate::matrix_trait::VectorCol;
use crate::{matrix_trait::MatrixTrait as _, Matrix};

use core::panic;

use core::clone::Clone;

impl<const ROWS: usize, const COLS: usize> Index<usize> for Matrix<ROWS, COLS> {
    type Output = Vec<f64, COLS>;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= ROWS {
            panic!("Row index is out of bounds");
        }
        &self.data[index]
    }
}

impl<const ROWS: usize, const COLS: usize> IndexMut<usize> for Matrix<ROWS, COLS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= ROWS {
            panic!("Row index is out of bounds");
        }
        &mut self.data[index]
    }
}

impl<const ROWS: usize, const COLS: usize> Add<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;

    fn add(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        let mut result: Matrix<ROWS, COLS> = Matrix::new().unwrap();

        for i in 0..ROWS {
            for j in 0..COLS {
                result[i][j] = self[i][j] + rhs[i][j];
            }
        }

        result
    }
}

impl<const ROWS: usize, const COLS: usize> AddAssign<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    fn add_assign(&mut self, rhs: Matrix<ROWS, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self[i][j] += rhs[i][j];
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Sub<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;

    fn sub(mut self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        for i in 0..ROWS {
            for j in 0..COLS {
                self[i][j] -= rhs[i][j];
            }
        }
        self
    }
}

impl<const ROWS: usize, const COLS: usize> SubAssign<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    fn sub_assign(&mut self, rhs: Matrix<ROWS, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self[i][j] -= rhs[i][j]
            }
        }
    }
}

impl<const LHS_ROWS: usize, const LHS_COLS: usize, const RHS_COLS: usize>
    Mul<Matrix<LHS_COLS, RHS_COLS>> for Matrix<LHS_ROWS, LHS_COLS>
{
    type Output = Matrix<LHS_ROWS, RHS_COLS>;

    fn mul(self, rhs: Matrix<LHS_COLS, RHS_COLS>) -> Self::Output {
        let mut result: Matrix<LHS_ROWS, RHS_COLS> = Matrix::new().unwrap();

        for i in 0..LHS_ROWS {
            for j in 0..RHS_COLS {
                let mut res = 0.;
                for k in 0..LHS_COLS {
                    res += self[i][k] * rhs[k][j];
                }
                result[i][j] = res;
            }
        }
        result
    }
}

///
/// Function that returns the absolute value of a f64 number
///
fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

///
/// Function that checks the equality between two f64 numbers with the tolerance of epsilon
///
pub(crate) fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
    abs(a - b) < epsilon
}

impl<const ROWS: usize, const COLS: usize> PartialEq<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    fn eq(&self, other: &Matrix<ROWS, COLS>) -> bool {
        for i in 0..ROWS {
            for j in 0..COLS {
                if !approx_equal(self[i][j], other[i][j], 1e-5) {
                    return false;
                }
            }
        }
        true
    }
}

impl<const N: usize> MulAssign<Matrix<N, N>> for Matrix<N, N> {
    fn mul_assign(&mut self, rhs: Matrix<N, N>) {
        let copy = self.clone();
        for i in 0..N {
            for j in 0..N {
                let mut result = 0.;
                for k in 0..N {
                    result += copy[i][k] * rhs[k][j];
                }
                self[i][j] = result;
            }
        }
    }
}

impl<const N: usize> MulAssign<&Matrix<N, N>> for Matrix<N, N> {
    fn mul_assign(&mut self, rhs: &Matrix<N, N>) {
        let copy = self.clone();
        for i in 0..N {
            for j in 0..N {
                let mut result = 0.;
                for k in 0..N {
                    result += copy[i][k] * rhs[k][j];
                }
                self[i][j] = result;
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize> MulAssign<f64> for Matrix<ROWS, COLS> {
    fn mul_assign(&mut self, rhs: f64) {
        for vec in self.iter_mut() {
            for elem in vec.iter_mut() {
                *elem *= rhs;
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<f64> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut copy = self.clone();
        copy *= rhs;
        copy
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<Matrix<ROWS, COLS>> for f64 {
    type Output = Matrix<ROWS, COLS>;

    fn mul(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        rhs * (self)
    }
}
