use crate::Matrix;
use core::marker::Sized;
use core::result::Result;
#[allow(dead_code)]
pub trait MatrixTrait<const ROWS: usize, const COLS: usize> {
    type TransposeType: MatrixTrait<COLS, ROWS>;

    ///
    /// Function that creates an matrix where all elements are equal to zero
    ///
    fn new() -> Result<Self, &'static str>
    where
        Self: Sized;

    ///
    /// Function that creates an identity matrix
    ///
    fn eye() -> Result<Self, &'static str>
    where
        Self: Sized;
    ///
    /// Function that creates an heapless matrix from a 2D array
    ///
    fn from_vector(data: [[f64; COLS]; ROWS]) -> Result<Self, &'static str>
    where
        Self: Sized;

    ///
    /// Function that transposes a matrix
    ///
    fn transpose(&self) -> Self::TransposeType;

    ///
    /// Function that converts a 1x1 matrix into a f64
    ///
    fn to_double(&self) -> Result<f64, &'static str>;

    ///
    /// Function that swaps two rows of a matrix
    ///  - @return - Err if index out of bounds
    fn swap_rows(&mut self, row1: usize, row2: usize) -> Result<(), &'static str>;

    ///
    /// function that swaps two columns of a matrix
    ///  - @return - Err if index out of bounds
    ///
    fn swap_cols(&mut self, col1: usize, col2: usize) -> Result<(), &'static str>;

    ///
    /// Function that returns the submatrix of a matrix
    ///  -  NEW_ROWS: usize - row dimensions of the new matrix
    ///  -  NEW_COLS: usize - column dimensions of the new matrix
    ///  -  row_start: usize - row starting index of the submatrix in the origin
    ///  -  col_start: usize - colzmn starting index of the submatrix in the origin
    ///
    ///  - @return - Err if the dimensions of the submatrix are invalid
    fn sub_matrix<const NEW_ROWS: usize, const NEW_COLS: usize>(
        &self,
        row_start: usize,
        col_start: usize,
    ) -> Result<Matrix<NEW_ROWS, NEW_COLS>, &'static str>;

    ///
    /// Function that takes the element from the array and sort them in a matrix row of dimensions ROWS x 1
    fn vector_to_row(elems: [f64; ROWS]) -> Result<Matrix<ROWS, 1>, &'static str>;

    ///
    /// Function that implements Moore-Penrose pseudoinversion
    ///
    ///  - DOUBLE: usize - must be equal to 2 * COLS if ROWS > COLS (left pseudoinversion), or 2 * ROWS if COLS > ROWS (right pseudoinversion)
    ///
    fn pinv<const DOUBLE: usize>(&self) -> Result<Matrix<COLS, ROWS>, &'static str>;
}

pub trait MatrixConcat<const ROWS: usize, const COLS: usize> {
    ///
    /// Function that concatenates matrices horizontally
    ///  - NEW_COLS: usize -  must be equal to COLS + RHS_COLS
    ///
    fn x_concat<const RHS_COLS: usize, const NEW_COLS: usize>(
        self,
        rhs: Matrix<ROWS, RHS_COLS>,
    ) -> Result<Matrix<ROWS, NEW_COLS>, &'static str>;

    ///
    /// Function that concatenates matrices vertically
    ///  - NEW_ROWS: usize - must be equal to ROWS + RHS_ROWS
    ///
    fn y_concat<const RHS_ROWS: usize, const NEW_ROWS: usize>(
        self,
        rhs: Matrix<RHS_ROWS, COLS>,
    ) -> Result<Matrix<NEW_ROWS, COLS>, &'static str>;
}
pub trait SquareMatrix<const N: usize> {
    ///
    /// Function that calculates the determinant of a square matrix
    ///
    fn det(&self) -> f64;

    ///
    /// Function that calculates the inversion of a sqaure matrix if possible, othervise it returns an Err
    ///  - DOUBLE_COLS: usize - must be equal to 2 * N dou to compiler restrictions
    ///
    fn inv<const DOUBLE_COLS: usize>(&self) -> Result<Matrix<N, N>, &'static str>;

    ///
    /// Function that calculates the pow of a square matrix
    ///
    fn pow(&self, n: usize) -> Matrix<N, N>;

    ///
    /// Function that takes the element from an array an makes a diagonal matrix using them
    ///
    fn diag(elems: [f64; N]) -> Result<Matrix<N, N>, &'static str>;
}

pub trait VectorCol<const ROWS: usize> {
    ///
    /// Function that inserts new data at the begginig of the matrix, and shifts other data by one to the end
    ///
    fn shift_data(&mut self, data: f64);
}

pub trait IsSquareMatrix {}

pub trait IsVectorCol {}
