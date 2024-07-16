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
/// Generic parameters:
/// * `const N: usize`: number of variables in regression
#[derive(Clone, Debug)]
pub struct MillerUpdatingRegression<const N: usize> {
    diagonal_matrix: [f64; N],
    upper_triangular_matrix: [[f64; N]; N],
    scaled_projections: [f64; N],
    num_observations: usize,
    sum_squared_errors: f64,
    sum_y: f64,
    sum_squared_y: f64,
    zero_tolerance: f64,
}

impl<const N: usize> MillerUpdatingRegression<N> {
    /// Construct en empty instance
    ///
    /// Parameters:
    /// * `zero_tolerance: f64`: threshold for machine zero
    pub fn empty(zero_tolerance: f64) -> MillerUpdatingRegression<N> {
        MillerUpdatingRegression {
            diagonal_matrix: [0.0; N],
            upper_triangular_matrix: [[0.0; N]; N],
            scaled_projections: [0.0; N],
            num_observations: 0,
            sum_squared_errors: 0.0,
            sum_y: 0.0,
            sum_squared_y: 0.0,
            zero_tolerance: zero_tolerance.abs(),
        }
    }

    /// Add an observation to the regression model
    ///
    /// Parameters:
    /// * `x: [f64; N]`: regressor values
    /// * `y: f64`: dependent variable
    pub fn add_observation(&mut self, x: [f64; N], y: f64) {
        self.include(x, 1.0, y);
        self.num_observations += 1
    }

    /// Update the QR decomposition with the new observation.
    ///
    /// Parameters:
    /// * `x: [f64; N]`: regressors
    /// * `w: f64`: weight
    /// * `y: f64`: regressand
    fn include(&mut self, mut x: [f64; N], mut w: f64, mut y: f64) {
        self.sum_y += y;
        self.sum_squared_y += y * y;

        for i in 0..N {
            if w.abs() < self.zero_tolerance {
                return;
            }

            if x[i].abs() < self.zero_tolerance {
                continue;
            }

            let wxi = w * x[i];
            let dpi = self.diagonal_matrix[i] + wxi * x[i];

            for k in i + 1..N {
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
    }

    /// Conduct a regression on the data in the model
    pub fn regress(&self) -> RegressionResult<N> {
        if self.num_observations <= N {
            // NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS("not enough data ({0} rows) for this many predictors ({1} predictors)"),
        }

        let mut regression = self.clone();
        let sqrt_diagonal_matrix: [f64; N] = core::array::from_fn(|i| regression.diagonal_matrix[i].sqrt());

        // calculate tolerances for singularity testing
        let tolerances: [f64; N] = core::array::from_fn(|col| {
            let total = (0..col).fold(sqrt_diagonal_matrix[col], |acc, row| {
                acc + regression.upper_triangular_matrix[row][col].abs() * sqrt_diagonal_matrix[row]
            });
            regression.zero_tolerance * total
        });

        // check for singularities and eliminate the offending columns
        let linear_dependencies: [bool; N] = core::array::from_fn(|col| {
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
                if col < N - 1 {
                    let x: [f64; N] = core::array::from_fn(|i| {
                        let value = regression.upper_triangular_matrix[col][i];
                        regression.upper_triangular_matrix[col][i] = 0.0;
                        value
                    });

                    let weight = regression.diagonal_matrix[col];
                    regression.diagonal_matrix[col] = 0.0;

                    let y = regression.scaled_projections[col];
                    regression.scaled_projections[col] = 0.0;

                    regression.include(x, weight, y);
                } else {
                    regression.sum_squared_errors += regression.diagonal_matrix[col]
                        * regression.scaled_projections[col]
                        * regression.scaled_projections[col];
                }

                true
            } else {
                false
            }
        });

        let valid_indices = |range: std::ops::Range<usize>| {
            range.filter(|i| !linear_dependencies[*i])
        };

        // conduct the linear regression and extract the parameter vector
        let parameters = {
            let mut beta = [f64::NAN; N];
            for i in valid_indices(0..N).rev() {
                beta[i] = valid_indices(i + 1..N).fold(self.scaled_projections[i], |sum, j| {
                    sum - self.upper_triangular_matrix[i][j] * beta[j]
                })
            }
            beta
        };

        let rank = valid_indices(0..N).count() as u32;

        // calculate the variance-covariance matrix
        let covariance = {
            let variance = regression.sum_squared_errors / (regression.num_observations as f64 - rank as f64);

            let mut inverted_upper_triangular_matrix = [[f64::NAN; N]; N];
            for col in valid_indices(1..N) {
                for row in valid_indices(0..col).rev() {
                    let total: f64 = valid_indices(row + 1..col).map(|k| {
                        -self.upper_triangular_matrix[row][k] * inverted_upper_triangular_matrix[k][col]
                    }).sum();
                    inverted_upper_triangular_matrix[row][col] = total - self.upper_triangular_matrix[row][col];
                }
            }

            let mut covmat = [[f64::NAN; N]; N];
            for row in valid_indices(0..N) {
                for col in valid_indices(row..N) {
                    let subtotal: f64 = valid_indices(col + 1..N).map(|k| {
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
                    covmat[col][row] = total * variance;
                }
            }
            covmat
        };

        RegressionResult {
            parameters,
            covariance,
            num_observations: regression.num_observations,
            rank,
            sum_y: regression.sum_y,
            sum_squared_y: regression.sum_squared_y,
            sum_squared_errors: regression.sum_squared_errors,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegressionResult<const N: usize> {
    pub parameters: [f64; N],
    pub covariance: [[f64; N]; N],
    pub num_observations: usize,
    pub rank: u32,
    pub sum_y: f64,
    pub sum_squared_y: f64,
    pub sum_squared_errors: f64,
}

impl<const N: usize> RegressionResult<N> {
    pub fn standard_error(&self) -> [f64; N] {
        core::array::from_fn(|i| {
            let var = self.covariance[i][i];
            if !var.is_nan() && var > f64::MIN {
                var.sqrt()
            } else {
                f64::NAN
            }
        })
    }

    pub fn mean_squared_error(&self) -> f64 {
        self.sum_squared_errors / (self.num_observations as f64 - self.rank as f64)
    }

    pub fn sum_of_squares_total(&self, has_constant: bool) -> f64 {
        if has_constant {
            self.sum_squared_y - self.sum_y * self.sum_y / self.num_observations as f64
        } else {
            self.sum_squared_y
        }
    }

    pub fn r_squared(&self, has_constant: bool) -> f64 {
        1.0 - self.sum_squared_errors / self.sum_of_squares_total(has_constant)
    }

    pub fn adjusted_r_squared(&self, has_constant: bool) -> f64 {
        if has_constant {
            1.0 - (self.sum_squared_errors * (self.num_observations as f64 - 1.0))
                / (self.sum_of_squares_total(has_constant) * (self.num_observations as f64 - self.rank as f64))
        } else {
            let r_squared = self.r_squared(has_constant);
            1.0 - (1.0 - r_squared) * (self.num_observations as f64 / (self.num_observations as f64 - self.rank as f64))
        }
    }
}
