# leastsquares

This library contains an implementation of the least squares method that allows for dynamic updating of the data, i.e. as observations become available, they can be added to the model without loading the entire data set in memory.

This is a rewrite of [MillerUpdatingRegression](https://github.com/apache/commons-math/blob/MATH_3_6_1/src/main/java/org/apache/commons/math3/stat/regression/MillerUpdatingRegression.java) from Java to Rust.

Example:
```
let mut model = MillerUpdatingRegression::empty(3, true, f64::EPSILON);

let x1 = [0.0, 1.0, 2.0];
let y1 = 3.0;
model.add_observation(&x1, y1)?;


let x2 = [4.0, 5.0, 6.0];
let y2 = 7.0;
model.add_observation(&x2, y2)?;

let result = model.regress()?;
println!("{:?}", result.parameters);
println!("{}", result.mean_squared_error());
```
