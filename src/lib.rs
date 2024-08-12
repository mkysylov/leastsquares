// Copyright 2024 Maksym Kysylov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(test)]
mod tests;

/// Miller Updating Regression
///
/// Ported from Apache Commons Math:
/// [org.apache.commons.math3.stat.regression.MillerUpdatingRegression](https://github.com/apache/commons-math/blob/MATH_3_6_1/src/main/java/org/apache/commons/math3/stat/regression/MillerUpdatingRegression.java)
///
/// The algorithm is described in:
/// Algorithm AS 274: Least Squares Routines to Supplement Those of Gentleman
/// Author(s): Alan J. Miller
/// Source: Journal of the Royal Statistical Society.
/// Series C (Applied Statistics), Vol. 41, No. 2
/// (1992), pp. 458-478
/// Published by: Blackwell Publishing for the Royal Statistical Society
/// Stable URL: http://www.jstor.org/stable/2347583
///
#[derive(Clone, Debug)]
pub struct MillerUpdatingRegression {
    diagonal_matrix: Box<[f64]>,
    upper_triangular_matrix: UTMat<f64>,
    scaled_projections: Box<[f64]>,
    num_observations: usize,
    sum_squared_errors: f64,
    sum_y: f64,
    sum_squared_y: f64,
    has_constant: bool,
    zero_tolerance: f64,
}

impl MillerUpdatingRegression {
    /// Construct en empty instance
    ///
    /// Parameters:
    /// * `num_variables: usize`: number of regressors, not including the constant
    /// * `include_constant: bool`: whether to include the intercept
    /// * `zero_tolerance: f64`: threshold for machine zero
    pub fn empty(num_variables: usize,
                 include_constant: bool,
                 zero_tolerance: f64) -> MillerUpdatingRegression {
        let size = num_variables + include_constant as usize;

        MillerUpdatingRegression {
            diagonal_matrix: vec![0.0; size].into_boxed_slice(),
            upper_triangular_matrix: UTMat::fill(0.0, size),
            scaled_projections: vec![0.0; size].into_boxed_slice(),
            num_observations: 0,
            sum_squared_errors: 0.0,
            sum_y: 0.0,
            sum_squared_y: 0.0,
            has_constant: include_constant,
            zero_tolerance: zero_tolerance.abs(),
        }
    }

    /// Add an observation to the regression model
    ///
    /// Parameters:
    /// * `x: &[f64]`: regressor values
    /// * `y: f64`: dependent variable
    pub fn add_observation(&mut self, x: &[f64], y: f64) -> Result<(), RegressionError> {
        let mut regressors: Vec<f64> = if self.has_constant { vec![1.0] } else { vec![] };
        regressors.extend(x);
        match self.include(&mut regressors, 1.0, y) {
            ok @ Ok(_) => {
                self.num_observations += 1;
                ok
            }
            error @ Err(_) => error,
        }
    }

    /// Update the QR decomposition with the new observation.
    ///
    /// Parameters:
    /// * `x: &mut [f64]`: regressors
    /// * `w: f64`: weight
    /// * `y: f64`: regressand
    fn include(&mut self, x: &mut [f64], mut w: f64, mut y: f64) -> Result<(), RegressionError> {
        let num_regressors = self.diagonal_matrix.len();
        if x.len() != num_regressors {
            return Err(RegressionError::DimensionMismatch { expected: num_regressors, actual: x.len() });
        }

        self.sum_y += y;
        self.sum_squared_y += y * y;

        for i in 0..num_regressors {
            if w.abs() < self.zero_tolerance {
                return Ok(());
            }

            if x[i].abs() < self.zero_tolerance {
                continue;
            }

            let wxi = w * x[i];
            let dpi = self.diagonal_matrix[i] + wxi * x[i];

            for k in i + 1..num_regressors {
                let xk_prev = x[k];
                x[k] -= x[i] * self.upper_triangular_matrix[i][k];
                self.upper_triangular_matrix[i][k] = (self.diagonal_matrix[i] * self.upper_triangular_matrix[i][k] + wxi * xk_prev) / dpi;
            }

            let y_prev = y;
            y -= x[i] * self.scaled_projections[i];
            self.scaled_projections[i] = (self.diagonal_matrix[i] * self.scaled_projections[i] + wxi * y_prev) / dpi;

            w *= self.diagonal_matrix[i] / dpi;
            self.diagonal_matrix[i] = dpi;
        }

        self.sum_squared_errors += w * y * y;
        Ok(())
    }

    /// Conduct a regression on the data in the model
    pub fn regress(&self) -> Result<RegressionResult, RegressionError> {
        let num_regressors = self.diagonal_matrix.len();
        if self.num_observations <= num_regressors {
            return Err(RegressionError::NotEnoughData { minimum: num_regressors, actual: self.num_observations });
        }

        let mut regression = self.clone();
        let sqrt_diagonal_matrix: Vec<f64> = regression.diagonal_matrix.iter()
            .map(|x| x.sqrt())
            .collect();

        // calculate tolerances for singularity testing
        let tolerances: Vec<f64> = (0..num_regressors).map(|col| {
            let total = (0..col).fold(sqrt_diagonal_matrix[col], |acc, row| {
                acc + regression.upper_triangular_matrix[row][col].abs() * sqrt_diagonal_matrix[row]
            });
            regression.zero_tolerance * total
        }).collect();

        // check for singularities and eliminate the offending columns
        let linear_dependencies: Vec<bool> = (0..num_regressors).map(|col| {
            let tolerance = tolerances[col];

            // set elements within R to zero if they are less than `tolerance` in absolute value
            // after being scaled by the square root of their row multiplier
            for row in 0..usize::max(col, 1) - 1 {
                if regression.upper_triangular_matrix[row][col].abs() * sqrt_diagonal_matrix[row] < tolerance {
                    regression.upper_triangular_matrix[row][col] = 0.0;
                }
            }

            // if diagonal element is near zero, set it to zero, use `include` to augment the
            // projections in the lower rows of the orthogonalization, and return an appropriate
            // linear dependency flag
            if sqrt_diagonal_matrix[col] < tolerance {
                if col < num_regressors - 1 {
                    let mut x = vec![0.0; num_regressors];
                    for i in col + 1..num_regressors {
                        x[i] = regression.upper_triangular_matrix[col][i];
                        regression.upper_triangular_matrix[col][i] = 0.0;
                    }

                    let weight = regression.diagonal_matrix[col];
                    regression.diagonal_matrix[col] = 0.0;

                    let y = regression.scaled_projections[col];
                    regression.scaled_projections[col] = 0.0;

                    regression.include(&mut x, weight, y).unwrap();
                } else {
                    regression.sum_squared_errors += regression.diagonal_matrix[col]
                        * regression.scaled_projections[col]
                        * regression.scaled_projections[col];
                }

                true
            } else {
                false
            }
        }).collect();

        let valid_indices = |range: std::ops::Range<usize>| {
            range.filter(|i| !linear_dependencies[*i])
        };

        // conduct the linear regression and extract the parameter vector
        let parameters = {
            let mut beta = vec![f64::NAN; num_regressors];
            for i in valid_indices(0..num_regressors).rev() {
                beta[i] = valid_indices(i + 1..num_regressors)
                    .fold(self.scaled_projections[i], |sum, j| {
                        sum - self.upper_triangular_matrix[i][j] * beta[j]
                    })
            }
            beta.into_boxed_slice()
        };

        let rank = valid_indices(0..num_regressors).count() as u32;

        // calculate the variance-covariance matrix
        let covariance = {
            let variance = regression.sum_squared_errors / (regression.num_observations as f64 - rank as f64);

            let mut inverted_upper_triangular_matrix = UTMat::fill(f64::NAN, num_regressors);
            for col in valid_indices(1..num_regressors) {
                for row in valid_indices(0..col).rev() {
                    let total: f64 = valid_indices(row + 1..col).map(|k| {
                        -self.upper_triangular_matrix[row][k] * inverted_upper_triangular_matrix[k][col]
                    }).sum();
                    inverted_upper_triangular_matrix[row][col] = total - self.upper_triangular_matrix[row][col];
                }
            }

            let mut covmat = UTMat::fill(f64::NAN, num_regressors);
            for row in valid_indices(0..num_regressors) {
                for col in valid_indices(row..num_regressors) {
                    let subtotal: f64 = valid_indices(col + 1..num_regressors).map(|k| {
                        inverted_upper_triangular_matrix[row][k]
                            * inverted_upper_triangular_matrix[col][k]
                            / self.diagonal_matrix[k]
                    }).sum();

                    let total = subtotal + if row == col {
                        1.0 / self.diagonal_matrix[col]
                    } else {
                        inverted_upper_triangular_matrix[row][col] / self.diagonal_matrix[col]
                    };

                    covmat[row][col] = total * variance;
                }
            }
            covmat
        };

        Ok(RegressionResult {
            parameters,
            covariance,
            num_observations: regression.num_observations,
            rank,
            sum_y: regression.sum_y,
            sum_squared_y: regression.sum_squared_y,
            sum_squared_errors: regression.sum_squared_errors,
            has_constant: regression.has_constant,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RegressionError {
    DimensionMismatch { expected: usize, actual: usize },
    NotEnoughData { minimum: usize, actual: usize },
}

#[derive(Clone, Debug)]
pub struct RegressionResult {
    pub parameters: Box<[f64]>,
    covariance: UTMat<f64>,
    pub num_observations: usize,
    pub rank: u32,
    pub sum_y: f64,
    pub sum_squared_y: f64,
    pub sum_squared_errors: f64,
    pub has_constant: bool,
}

impl RegressionResult {
    pub fn covariance(&self, i: usize, j: usize) -> f64 {
        if i > j {
            self.covariance(j, i)
        } else {
            self.covariance[i][j]
        }
    }
    pub fn standard_error(&self) -> Box<[f64]> {
        (0..self.parameters.len())
            .map(|i| {
                let var = self.covariance(i, i);
                if !var.is_nan() && var > f64::MIN {
                    var.sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    pub fn mean_squared_error(&self) -> f64 {
        self.sum_squared_errors / (self.num_observations as f64 - self.rank as f64)
    }

    pub fn sum_of_squares_total(&self) -> f64 {
        if self.has_constant {
            self.sum_squared_y - self.sum_y * self.sum_y / self.num_observations as f64
        } else {
            self.sum_squared_y
        }
    }

    pub fn r_squared(&self) -> f64 {
        1.0 - self.sum_squared_errors / self.sum_of_squares_total()
    }

    pub fn adjusted_r_squared(&self) -> f64 {
        if self.has_constant {
            1.0 - (self.sum_squared_errors * (self.num_observations as f64 - 1.0))
                / (self.sum_of_squares_total() * (self.num_observations as f64 - self.rank as f64))
        } else {
            let r_squared = self.r_squared();
            1.0 - (1.0 - r_squared) * (self.num_observations as f64 / (self.num_observations as f64 - self.rank as f64))
        }
    }
}

/// Upper-triangular matrix
#[derive(Clone, Debug)]
struct UTMat<T>(Box<[T]>);

impl<T: Clone> UTMat<T> {
    pub fn fill(value: T, size: usize) -> UTMat<T> {
        UTMat(vec![value; size * (size + 1) / 2].into_boxed_slice())
    }

    pub fn size(&self) -> usize {
        ((((self.0.len() * 8 + 1) as f64).sqrt() - 1.0) / 2.0) as usize
    }

    fn row_index(&self, row: usize) -> usize {
        row * self.size() - row * (row + 1) / 2
    }
}

impl<T: Clone> std::ops::Index<usize> for UTMat<T> {
    type Output = [T];

    fn index(&self, row: usize) -> &Self::Output {
        let offset = self.row_index(row);
        &self.0[offset..]
    }
}

impl<T: Clone> std::ops::IndexMut<usize> for UTMat<T> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let offset = self.row_index(row);
        &mut self.0[offset..]
    }
}
