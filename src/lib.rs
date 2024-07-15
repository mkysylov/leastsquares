/// Miller Updating Regression
///
/// * `N`: number of variables in regression
#[derive(Clone, Debug)]
pub struct MillerUpdatingRegression<const N: usize> {
    /// diagonals of cross products matrix
    d: [f64; N],
    /// the elements of the R`Y
    rhs: [f64; N],
    /// the off diagonal portion of the R matrix
    r: [[f64; N]; N],
    /// the tolerance for each of the variables
    tol: [f64; N],
    /// residual sum of squares for all nested regressions
    rss: [f64; N],
    /// number of observations entered
    nobs: usize,
    /// sum of squared errors of largest regression
    sserr: f64,
    /// has rss been called
    rssSet: bool,
    /// has the tolerance setting method been called
    tolSet: bool,
    /// flags for variables with linear dependency problems
    lindep: [bool; N],
    /// summation of Y variable
    sumy: f64,
    /// summation of squared Y values
    sumsqy: f64,
    /// zero tolerance
    epsilon: f64,
}

impl<const N: usize> MillerUpdatingRegression<N> {
    /// Construct en empty instance
    ///
    /// * `error_tolerance`:  zero tolerance, how machine zero is determined
    fn empty(error_tolerance: f64) -> MillerUpdatingRegression<N> {
        MillerUpdatingRegression {
            d: [0f64; N],
            rhs: [0f64; N],
            r: [[0f64; N]; N],
            tol: [0f64; N],
            rss: [0f64; N],
            nobs: 0,
            sserr: 0f64,
            rssSet: false,
            tolSet: false,
            lindep: [false; N],
            sumy: 0f64,
            sumsqy: 0f64,
            epsilon: error_tolerance.abs(),
        }
    }

    /**
     * Adds an observation to the regression model.
     * @param x the array with regressor values
     * @param y  the value of dependent variable given these regressors
     * @exception ModelSpecificationException if the length of {@code x} does not equal
     * the number of independent variables in the model
     */
    pub fn add_observation(&mut self, x: [f64; N], y: f64) {
        self.include(x, 1.0, y);
        self.nobs += 1
    }

    /**
     * The include method is where the QR decomposition occurs. This statement forms all
     * intermediate data which will be used for all derivative measures.
     * According to the miller paper, note that in the original implementation the x vector
     * is overwritten. In this implementation, the include method is passed a copy of the
     * original data vector so that there is no contamination of the data. Additionally,
     * this method differs slightly from Gentleman's method, in that the assumption is
     * of dense design matrices, there is some advantage in using the original gentleman algorithm
     * on sparse matrices.
     *
     * @param x observations on the regressors
     * @param w weight of the this observation (-1,1)
     * @param y observation on the regressand
     */
    fn include(&mut self, mut x: [f64; N], mut w: f64, mut y: f64) {
        let mut dpi = 0f64;
        let mut xk = 0f64;
        self.rssSet = false;
        self.sumy = Self::smartAdd(y, self.sumy);
        self.sumsqy = Self::smartAdd(self.sumsqy, y * y);

        for i in 0..N {
            if w == 0.0 {
                return;
            }
            let xi = x[i];

            if xi == 0.0 {
                continue;
            }
            let di = self.d[i];
            let wxi = w * xi;
            let wPrev = w;
            if di != 0.0 {
                dpi = Self::smartAdd(di, wxi * xi);
                let tmp = wxi * xi / di;
                if tmp.abs() > f64::EPSILON {
                    w = (di * w) / dpi;
                }
            } else {
                dpi = wxi * xi;
                w = 0.0;
            }
            self.d[i] = dpi;
            for k in i + 1..N {
                xk = x[k];
                x[k] = Self::smartAdd(xk, -xi * self.r[i][k]);
                if di != 0.0 {
                    self.r[i][k] = Self::smartAdd(di * self.r[i][k], (wPrev * xi) * xk) / dpi;
                } else {
                    self.r[i][k] = xk / xi;
                }
            }
            xk = y;
            y = Self::smartAdd(xk, -xi * self.rhs[i]);
            if di != 0.0 {
                self.rhs[i] = Self::smartAdd(di * self.rhs[i], wxi * xk) / dpi;
            } else {
                self.rhs[i] = xk / xi;
            }
        }
        self.sserr = Self::smartAdd(self.sserr, w * y * y);
    }

    /**
     * Adds to number a and b such that the contamination due to
     * numerical smallness of one addend does not corrupt the sum.
     * @param a - an addend
     * @param b - an addend
     * @return the sum of the a and b
     */
    fn smartAdd(a: f64, b: f64) -> f64 {
        let aa = a.abs();
        let ba = b.abs();
        if aa > ba {
            let eps = aa * f64::EPSILON;
            if ba > eps {
                return a + b;
            }
            return a;
        } else {
            let eps = ba * f64::EPSILON;
            if aa > eps {
                return a + b;
            }
            return b;
        }
    }

    pub fn regress(&mut self) -> RegressionResult<N> {
        if self.nobs <= N {
            // NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS("not enough data ({0} rows) for this many predictors ({1} predictors)"),
        }

        self.tolset();
        self.singcheck();
        let beta = self.regcf();

        self.ss();

        let cov = self.cov();

        let mut rnk = 0;
        for i in 0..N {
            if !self.lindep[i] {
                rnk += 1;
            }
        }

        RegressionResult {
            parameters: beta,
            varcov: cov,
            nobs: self.nobs,
            rank: rnk,
            sumy: self.sumy,
            sumysq: self.sumsqy,
            sse: self.sserr,
        }
    }

    /**
     * This sets up tolerances for singularity testing.
     */
    fn tolset(&mut self) {
        let work_tolset: [f64; N] = core::array::from_fn(|i| self.d[i].sqrt());
        self.tol[0] = self.epsilon * work_tolset[0];
        for col in 1..N {
            let mut total = work_tolset[col];
            for row in 0..col {
                total += self.r[row][col] * work_tolset[row];
            }
            self.tol[col] = self.epsilon * total;
        }
        self.tolSet = true;
    }

    /**
     * The method which checks for singularities and then eliminates the offending
     * columns.
     */
    fn singcheck(&mut self) {
        let work_sing: [f64; N] = core::array::from_fn(|i| self.d[i].sqrt());

        for col in 0..N {
            // Set elements within R to zero if they are less than tol(col) in
            // absolute value after being scaled by the square root of their row
            // multiplier
            let temp = self.tol[col];
            // for row in 0..col - 1 {
            for row in 0..usize::max(col, 1) - 1 {
                if self.r[row][col].abs() * work_sing[row] < temp {
                    self.r[row][col] = 0.0;
                }
            }
            // If diagonal element is near zero, set it to zero, set appropriate
            // element of LINDEP, and use INCLUD to augment the projections in
            // the lower rows of the orthogonalization.
            self.lindep[col] = false;
            if work_sing[col] < temp {
                self.lindep[col] = true;
                if col < N - 1 {
                    let mut xSing = [0f64; N];
                    for xi in col + 1..N {
                        xSing[xi] = self.r[col][xi];
                        self.r[col][xi] = 0.0;
                    }
                    let y = self.rhs[col];
                    let weight = self.d[col];
                    self.d[col] = 0.0;
                    self.rhs[col] = 0.0;
                    self.include(xSing, weight, y);
                } else {
                    self.sserr += self.d[col] * self.rhs[col] * self.rhs[col];
                }
            }
        }
    }

    /**
     * The regcf method conducts the linear regression and extracts the
     * parameter vector. Notice that the algorithm can do subset regression
     * with no alteration.
     *
     * @param nreq how many of the regressors to include (either in canonical
     * order, or in the current reordered state)
     * @return an array with the estimated slope coefficients
     * @throws ModelSpecificationException if {@code nreq} is less than 1
     * or greater than the number of independent variables
     */
    fn regcf(&mut self) -> [f64; N] {
        if !self.tolSet {
            self.tolset();
        }
        let mut ret = [0f64; N];
        let mut rankProblem = false;
        for i in (0..N).rev() {
            if self.d[i].sqrt() < self.tol[i] {
                ret[i] = 0.0;
                self.d[i] = 0.0;
                rankProblem = true;
            } else {
                ret[i] = self.rhs[i];
                for j in i + 1..N {
                    ret[i] = Self::smartAdd(ret[i], -self.r[i][j] * ret[j]);
                }
            }
        }
        if rankProblem {
            for i in 0..N {
                if self.lindep[i] {
                    ret[i] = f64::NAN;
                }
            }
        }
        return ret;
    }

    /**
     * Calculates the sum of squared errors for the full regression.
     * and all subsets in the following manner: <pre>
     * rss[] ={
     * ResidualSumOfSquares_allNvars,
     * ResidualSumOfSquares_FirstNvars-1,
     * ResidualSumOfSquares_FirstNvars-2,
     * ..., ResidualSumOfSquares_FirstVariable} </pre>
     */
    fn ss(&mut self) {
        let mut total = self.sserr;
        self.rss[N - 1] = self.sserr;
        for i in (1..N - 1).rev() {
            total += self.d[i] * self.rhs[i] * self.rhs[i];
            self.rss[i - 1] = total;
        }
        self.rssSet = true;
    }

    /**
     * Calculates the cov matrix assuming only the first nreq variables are
     * included in the calculation. The returned array contains a symmetric
     * matrix stored in lower triangular form. The matrix will have
     * ( nreq + 1 ) * nreq / 2 elements. For illustration <pre>
     * cov =
     * {
     *  cov_00,
     *  cov_10, cov_11,
     *  cov_20, cov_21, cov22,
     *  ...
     * } </pre>
     *
     * @param nreq how many of the regressors to include (either in canonical
     * order, or in the current reordered state)
     * @return an array with the variance covariance of the included
     * regressors in lower triangular form
     */
    fn cov(&self) -> [[f64; N]; N] {
        // if (this.nobs <= nreq) {
        //     return null;
        // }
        let mut rnk = 0.0;
        for i in 0..N {
            if !self.lindep[i] {
                rnk += 1.0;
            }
        }
        let var = self.rss[N - 1] / (self.nobs as f64 - rnk);
        let rinv = self.inverse();
        let mut covmat = [[f64::NAN; N]; N];
        for row in 0..N {
            if !self.lindep[row] {
                for col in row..N {
                    if !self.lindep[col] {
                        let mut total = if row == col {
                            1.0 / self.d[col]
                        } else {
                            rinv[row][col] / self.d[col]
                        };
                        for k in col + 1..N {
                            if !self.lindep[k] {
                                total += rinv[row][k] * rinv[col][k] / self.d[k];
                            }
                        }
                        covmat[col][row] = total * var;
                    }
                }
            }
        }

        covmat
    }

    /**
     * This internal method calculates the inverse of the upper-triangular portion
     * of the R matrix.
     * @param rinv  the storage for the inverse of r
     * @param nreq how many of the regressors to include (either in canonical
     * order, or in the current reordered state)
     */
    fn inverse(&self) -> [[f64; N]; N] {
        let mut rinv = [[f64::NAN; N]; N];
        for row in (1..N).rev() {
            if !self.lindep[row] {
                for col in (row + 1..=N).rev() {
                    let mut total = 0.0;
                    for k in row..col - 1 {
                        if !self.lindep[k] {
                            total += -self.r[row - 1][k] * rinv[k][col - 1]
                        }
                    }
                    rinv[row - 1][col - 1] = total - self.r[row - 1][col - 1]
                }
            }
        }

        rinv
    }
}

#[derive(Clone, Debug)]
pub struct RegressionResult<const N: usize> {
    pub parameters: [f64; N],
    pub varcov: [[f64; N]; N],
    pub nobs: usize,
    pub rank: i32,
    pub sumy: f64,
    pub sumysq: f64,
    pub sse: f64,
}

impl<const N: usize> RegressionResult<N> {
    pub fn stderr(&self) -> [f64; N] {
        let mut se = [f64::NAN; N];

        for i in 0..self.parameters.len() {
            let var = self.varcov[i][i];
            if !var.is_nan() && var > f64::MIN {
                se[i] = var.sqrt();
            }
        }
        se
    }

    pub fn mse(&self) -> f64 {
        self.sse / (self.nobs as f64 - self.rank as f64)
    }

    pub fn sst(&self, has_constant: bool) -> f64 {
        if has_constant {
            self.sumysq - self.sumy * self.sumy / self.nobs as f64
        } else {
            self.sumysq
        }
    }

    pub fn r_squared(&self, has_constant: bool) -> f64 {
        1.0 - self.sse / self.sst(has_constant)
    }

    pub fn adjusted_r_squared(&self, has_constant: bool) -> f64 {
        if has_constant {
            let sst = self.sst(has_constant);
            1.0 - (self.sse * (self.nobs as f64 - 1.0))
                / (sst * (self.nobs as f64 - self.rank as f64))
        } else {
            let r_squared = self.r_squared(has_constant);
            1.0 - (1.0 - r_squared) * (self.nobs as f64 / (self.nobs as f64 - self.rank as f64))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::MillerUpdatingRegression;

    #[test]
    fn regress_airline() {
        let data: [[f64; 90]; 6] = include!("datasets/air.in");
        let mut instance = MillerUpdatingRegression::<4>::empty(f64::EPSILON);

        let mut x = [[0.0; 4]; 90];
        let mut y = [0.0; 90];
        for i in 0..data[0].len() {
            x[i][0] = 1.0;
            x[i][1] = data[3][i].ln();
            x[i][2] = data[4][i].ln();
            x[i][3] = data[5][i];
            y[i] = data[2][i].ln();
        }

        for (xi, yi) in zip(x, y) {
            instance.add_observation(xi, yi);
        }
        let result = instance.regress();

        assert!(zip([9.5169, 0.8827, 0.4540, -1.6275], result.parameters)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-4));
        assert!(
            zip([0.2292445, 0.0132545, 0.0203042, 0.345302], result.stderr())
                .all(|(expected, actual)| (expected - actual).abs() < 1e-4)
        );
        assert!((0.01552839 - result.mse()).abs() < 1.0e-8);
        assert!((0.9883 - result.r_squared(true)).abs() < 1.0e-4);
    }

    #[test]
    fn filipelli() {
        let data: [[f64; 82]; 2] = include!("datasets/filipelli.in");

        let mut instance = MillerUpdatingRegression::<11>::empty(f64::EPSILON);
        for i in 0..data[0].len() {
            let x: [f64; 11] = core::array::from_fn(|j| data[1][i].powi(j as i32));
            let y = data[0][i];

            instance.add_observation(x, y);
        }

        let result = instance.regress();

        assert!(zip(
            [
                -1467.48961422980,
                -2772.17959193342,
                -2316.37108160893,
                -1127.97394098372,
                -354.478233703349,
                -75.1242017393757,
                -10.8753180355343,
                -1.06221498588947,
                -0.670191154593408E-01,
                -0.246781078275479E-02,
                -0.402962525080404E-04
            ],
            result.parameters
        )
        .all(|(expected, actual)| ((expected - actual) / expected).abs() < 1e-6));

        assert!(zip(
            [
                298.084530995537,
                559.779865474950,
                466.477572127796,
                227.204274477751,
                71.6478660875927,
                15.2897178747400,
                2.23691159816033,
                0.221624321934227,
                0.142363763154724E-01,
                0.535617408889821E-03,
                0.896632837373868E-05
            ],
            result.stderr()
        )
        .all(|(expected, actual)| ((expected - actual) / expected).abs() < 1e-6));

        assert!((0.996727416185620 - result.r_squared(true)).abs() < 1.0e-10);
        assert!((0.112091743968020E-04 - result.mse()).abs() < 1.0e-10);
        assert!((0.795851382172941E-03 - result.sse).abs() < 1.0e-10);
    }

    #[test]
    fn wampler1() {
        let data: [f64; 42] = [
            1.0, 0.0, 6.0, 1.0, 63.0, 2.0, 364.0, 3.0, 1365.0, 4.0, 3906.0, 5.0, 9331.0, 6.0,
            19608.0, 7.0, 37449.0, 8.0, 66430.0, 9.0, 111111.0, 10.0, 177156.0, 11.0, 271453.0,
            12.0, 402234.0, 13.0, 579195.0, 14.0, 813616.0, 15.0, 1118481.0, 16.0, 1508598.0, 17.0,
            2000719.0, 18.0, 2613660.0, 19.0, 3368421.0, 20.0,
        ];

        let mut instance = MillerUpdatingRegression::<6>::empty(f64::EPSILON);
        for i in 0..data.len() / 2 {
            let x: [f64; 6] = core::array::from_fn(|j| data[i * 2 + 1].powi(j as i32));
            instance.add_observation(x, data[i * 2]);
        }
        let result = instance.regress();

        assert!(zip([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], result.parameters)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!(zip([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], result.stderr())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!((1.0 - result.r_squared(true)).abs() < 1.0e-10);
        assert!((0.0 - result.mse()).abs() < 1.0e-8);
        assert!((0.0 - result.sse).abs() < 1.0e-8);
    }

    #[test]
    fn wampler2() {
        let data: [f64; 42] = [
            1.00000, 0.0, 1.11111, 1.0, 1.24992, 2.0, 1.42753, 3.0, 1.65984, 4.0, 1.96875, 5.0,
            2.38336, 6.0, 2.94117, 7.0, 3.68928, 8.0, 4.68559, 9.0, 6.00000, 10.0, 7.71561, 11.0,
            9.92992, 12.0, 12.75603, 13.0, 16.32384, 14.0, 20.78125, 15.0, 26.29536, 16.0,
            33.05367, 17.0, 41.26528, 18.0, 51.16209, 19.0, 63.00000, 20.0,
        ];

        let mut instance = MillerUpdatingRegression::<6>::empty(f64::EPSILON);
        for i in 0..data.len() / 2 {
            let x: [f64; 6] = core::array::from_fn(|j| data[i * 2 + 1].powi(j as i32));
            instance.add_observation(x, data[i * 2]);
        }
        let result = instance.regress();

        assert!(zip([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], result.parameters)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!(zip([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], result.stderr())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!((1.0 - result.r_squared(true)).abs() < 1.0e-10);
        assert!((0.0 - result.mse()).abs() < 1.0e-8);
        assert!((0.0 - result.sse).abs() < 1.0e-8);
    }

    #[test]
    fn wampler3() {
        let data: [f64; 42] = [
            760.0, 0.0, -2042.0, 1.0, 2111.0, 2.0, -1684.0, 3.0, 3888.0, 4.0, 1858.0, 5.0, 11379.0,
            6.0, 17560.0, 7.0, 39287.0, 8.0, 64382.0, 9.0, 113159.0, 10.0, 175108.0, 11.0,
            273291.0, 12.0, 400186.0, 13.0, 581243.0, 14.0, 811568.0, 15.0, 1121004.0, 16.0,
            1506550.0, 17.0, 2002767.0, 18.0, 2611612.0, 19.0, 3369180.0, 20.0,
        ];

        let mut instance = MillerUpdatingRegression::<6>::empty(f64::EPSILON);
        for i in 0..data.len() / 2 {
            let x: [f64; 6] = core::array::from_fn(|j| data[i * 2 + 1].powi(j as i32));
            instance.add_observation(x, data[i * 2]);
        }
        let result = instance.regress();

        assert!(zip([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], result.parameters)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!(zip(
            [
                2152.32624678170,
                2363.55173469681,
                779.343524331583,
                101.475507550350,
                5.64566512170752,
                0.112324854679312
            ],
            result.stderr()
        )
        .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!((0.999995559025820 - result.r_squared(true)).abs() < 1.0e-10);
        assert!((5570284.53333333 - result.mse()).abs() < 1.0e-7);
        assert!((83554268.0000000 - result.sse).abs() < 1.0e-6);
    }

    #[test]
    fn wampler4() {
        let data: [f64; 42] = [
            75901.0, 0.0, -204794.0, 1.0, 204863.0, 2.0, -204436.0, 3.0, 253665.0, 4.0, -200894.0,
            5.0, 214131.0, 6.0, -185192.0, 7.0, 221249.0, 8.0, -138370.0, 9.0, 315911.0, 10.0,
            -27644.0, 11.0, 455253.0, 12.0, 197434.0, 13.0, 783995.0, 14.0, 608816.0, 15.0,
            1370781.0, 16.0, 1303798.0, 17.0, 2205519.0, 18.0, 2408860.0, 19.0, 3444321.0, 20.0,
        ];

        let mut instance = MillerUpdatingRegression::<6>::empty(f64::EPSILON);
        for i in 0..data.len() / 2 {
            let x: [f64; 6] = core::array::from_fn(|j| data[i * 2 + 1].powi(j as i32));
            instance.add_observation(x, data[i * 2]);
        }
        let result = instance.regress();

        assert!(zip([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], result.parameters)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!(zip(
            [
                215232.624678170,
                236355.173469681,
                77934.3524331583,
                10147.5507550350,
                564.566512170752,
                11.2324854679312
            ],
            result.stderr()
        )
        .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!((0.957478440825662 - result.r_squared(true)).abs() < 1.0e-10);
        assert!((55702845333.3333 - result.mse()).abs() < 1.0e-4);
        assert!((835542680000.000 - result.sse).abs() < 1.0e-3);
    }

    #[test]
    fn longley_with_constant() {
        let data: [f64; 112] = include!("datasets/langley.in");

        let mut instance = MillerUpdatingRegression::<7>::empty(f64::EPSILON);
        for i in 0..data.len() / 7 {
            let x: [f64; 7] = core::array::from_fn(|j| if j == 0 { 1.0 } else { data[i * 7 + j] });
            instance.add_observation(x, data[i * 7]);
        }
        let result = instance.regress();

        assert!(zip(
            [
                -3482258.63459582,
                15.0618722713733,
                -0.358191792925910E-01,
                -2.02022980381683,
                -1.03322686717359,
                -0.511041056535807E-01,
                1829.15146461355
            ],
            result.parameters
        )
        .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
        assert!(zip(
            [
                890420.383607373,
                84.9149257747669,
                0.334910077722432E-01,
                0.488399681651699,
                0.214274163161675,
                0.226073200069370,
                455.478499142212
            ],
            result.stderr()
        )
        .all(|(expected, actual)| (expected - actual).abs() < 1e-6));
        assert!((0.995479004577296 - result.r_squared(true)).abs() < 1.0e-12);
        assert!((0.992465007628826 - result.adjusted_r_squared(true)).abs() < 1.0e-12);
    }

    #[test]
    fn longley_without_constant() {
        let data: [f64; 112] = include!("datasets/langley.in");

        let mut instance = MillerUpdatingRegression::<6>::empty(f64::EPSILON);
        for i in 0..data.len() / 7 {
            let x: [f64; 6] = core::array::from_fn(|j| data[i * 7 + 1 + j]);
            instance.add_observation(x, data[i * 7]);
        }
        let result = instance.regress();

        assert!(zip(
            [
                -52.99357013868291,
                0.07107319907358,
                -0.42346585566399,
                -0.57256866841929,
                -0.41420358884978,
                48.41786562001326
            ],
            result.parameters
        )
        .all(|(expected, actual)| (expected - actual).abs() < 1e-11));
        assert!(zip(
            [
                129.54486693117232,
                0.03016640003786,
                0.41773654056612,
                0.27899087467676,
                0.32128496193363,
                17.68948737819961
            ],
            result.stderr()
        )
        .all(|(expected, actual)| (expected - actual).abs() < 1e-11));
        assert!((0.9999670130706 - result.r_squared(false)).abs() < 1.0e-12);
        assert!((0.999947220913 - result.adjusted_r_squared(false)).abs() < 1.0e-12);
    }
}
