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
        zip([0.2292445, 0.0132545, 0.0203042, 0.345302], result.standard_error())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-4)
    );
    assert!((0.01552839 - result.mean_squared_error()).abs() < 1.0e-8);
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
        result.parameters,
    ).all(|(expected, actual)| ((expected - actual) / expected).abs() < 1e-6));

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
        result.standard_error(),
    ).all(|(expected, actual)| ((expected - actual) / expected).abs() < 1e-6));

    assert!((0.996727416185620 - result.r_squared(true)).abs() < 1.0e-10);
    assert!((0.112091743968020E-04 - result.mean_squared_error()).abs() < 1.0e-10);
    assert!((0.795851382172941E-03 - result.sum_squared_errors).abs() < 1.0e-10);
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
    assert!(zip([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], result.standard_error())
        .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
    assert!((1.0 - result.r_squared(true)).abs() < 1.0e-10);
    assert!((0.0 - result.mean_squared_error()).abs() < 1.0e-8);
    assert!((0.0 - result.sum_squared_errors).abs() < 1.0e-8);
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
    assert!(zip([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], result.standard_error())
        .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
    assert!((1.0 - result.r_squared(true)).abs() < 1.0e-10);
    assert!((0.0 - result.mean_squared_error()).abs() < 1.0e-8);
    assert!((0.0 - result.sum_squared_errors).abs() < 1.0e-8);
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
        result.standard_error(),
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-8));
    assert!((0.999995559025820 - result.r_squared(true)).abs() < 1.0e-10);
    assert!((5570284.53333333 - result.mean_squared_error()).abs() < 1.0e-7);
    assert!((83554268.0000000 - result.sum_squared_errors).abs() < 1.0e-6);
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
        result.standard_error(),
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-8));
    assert!((0.957478440825662 - result.r_squared(true)).abs() < 1.0e-10);
    assert!((55702845333.3333 - result.mean_squared_error()).abs() < 1.0e-4);
    assert!((835542680000.000 - result.sum_squared_errors).abs() < 1.0e-3);
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
        result.parameters,
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-8));
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
        result.standard_error(),
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-6));
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
        result.parameters,
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-11));
    assert!(zip(
        [
            129.54486693117232,
            0.03016640003786,
            0.41773654056612,
            0.27899087467676,
            0.32128496193363,
            17.68948737819961
        ],
        result.standard_error(),
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-11));
    assert!((0.9999670130706 - result.r_squared(false)).abs() < 1.0e-12);
    assert!((0.999947220913 - result.adjusted_r_squared(false)).abs() < 1.0e-12);
}

#[test]
fn redundant_column_1() {
    let data: [[f64; 90]; 6] = include!("datasets/air.in");
    let mut instance1 = MillerUpdatingRegression::<4>::empty(f64::EPSILON);
    let mut instance2 = MillerUpdatingRegression::<5>::empty(f64::EPSILON);

    let mut x1 = [[0.0; 4]; 90];
    let mut x2 = [[0.0; 5]; 90];
    let mut y = [0.0; 90];
    for i in 0..data[0].len() {
        x1[i] = [1.0, data[3][i].ln(), data[4][i].ln(), data[5][i]];
        x2[i] = [
            1.0,
            data[3][i].ln(),
            data[4][i].ln(),
            data[5][i],
            data[5][i],
        ];
        y[i] = data[2][i].ln();
    }

    for (xi, yi) in zip(x1, y) {
        instance1.add_observation(xi, yi);
    }
    let result1 = instance1.regress();

    for (xi, yi) in zip(x2, y) {
        instance2.add_observation(xi, yi);
    }
    let result2 = instance2.regress();

    assert!(zip(result1.parameters, result2.parameters)
        .all(|(expected, actual)| (expected - actual).abs() < 1e-8));
    assert!(result2.parameters[4].is_nan());
    assert!(
        (result1.adjusted_r_squared(true) - result2.adjusted_r_squared(true)).abs() < 1.0e-8
    );
    assert!((result1.sum_squared_errors - result2.sum_squared_errors).abs() < 1.0e-8);
    assert!((result1.mean_squared_error() - result2.mean_squared_error()).abs() < 1.0e-8);
    assert!((result1.r_squared(true) - result2.r_squared(true)).abs() < 1.0e-8);
}

#[test]
fn redundant_column_3() {
    let data: [[f64; 90]; 6] = include!("datasets/air.in");
    let mut instance1 = MillerUpdatingRegression::<4>::empty(f64::EPSILON);
    let mut instance2 = MillerUpdatingRegression::<7>::empty(f64::EPSILON);

    let mut x1 = [[0.0; 4]; 90];
    let mut x2 = [[0.0; 7]; 90];
    let mut y = [0.0; 90];
    for i in 0..data[0].len() {
        x1[i] = [1.0, data[3][i].ln(), data[4][i].ln(), data[5][i]];
        x2[i] = [
            1.0,
            1.0,
            data[3][i].ln(),
            data[4][i].ln(),
            data[3][i].ln(),
            data[5][i],
            data[4][i].ln(),
        ];

        y[i] = data[2][i].ln();
    }

    for (xi, yi) in zip(x1, y) {
        instance1.add_observation(xi, yi);
    }
    let result1 = instance1.regress();

    for (xi, yi) in zip(x2, y) {
        instance2.add_observation(xi, yi);
    }
    let result2 = instance2.regress();

    assert!(zip(
        result1.parameters,
        [
            result2.parameters[0],
            result2.parameters[2],
            result2.parameters[3],
            result2.parameters[5]
        ],
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-8));
    assert!(result2.parameters[1].is_nan());
    assert!(result2.parameters[4].is_nan());
    assert!(result2.parameters[6].is_nan());

    assert!(zip(
        result1.standard_error(),
        [
            result2.standard_error()[0],
            result2.standard_error()[2],
            result2.standard_error()[3],
            result2.standard_error()[5]
        ],
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-8));

    assert!(zip(
        [
            result1.covariance[0][0],
            result1.covariance[0][1],
            result1.covariance[0][2],
            result1.covariance[0][3],
            result1.covariance[1][0],
            result1.covariance[1][1],
            result1.covariance[1][2],
            result1.covariance[2][0],
            result1.covariance[2][1],
            result1.covariance[3][3]
        ],
        [
            result2.covariance[0][0],
            result2.covariance[0][2],
            result2.covariance[0][3],
            result2.covariance[0][5],
            result2.covariance[2][0],
            result2.covariance[2][2],
            result2.covariance[2][3],
            result2.covariance[3][0],
            result2.covariance[3][2],
            result2.covariance[5][5]
        ],
    ).all(|(expected, actual)| (expected - actual).abs() < 1e-8));

    assert!([
        result2.covariance[0][1],
        result2.covariance[0][4],
        result2.covariance[1][0],
        result2.covariance[1][1],
        result2.covariance[1][2],
        result2.covariance[1][3],
        result2.covariance[1][4],
        result2.covariance[2][1],
        result2.covariance[3][1],
        result2.covariance[4][0],
        result2.covariance[4][1],
    ].iter().all(|x| x.is_nan()));

    assert!(
        (result1.adjusted_r_squared(true) - result2.adjusted_r_squared(true)).abs() < 1.0e-8
    );
    assert!((result1.sum_squared_errors - result2.sum_squared_errors).abs() < 1.0e-8);
    assert!((result1.mean_squared_error() - result2.mean_squared_error()).abs() < 1.0e-8);
    assert!((result1.r_squared(true) - result2.r_squared(true)).abs() < 1.0e-8);
}