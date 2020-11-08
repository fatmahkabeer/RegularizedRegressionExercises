# RegularizedRegressionExercises

### Ridge Regression idea:
The idea behind Ridge Regression is to find a New Line that doesn't fit the Training Data as well. In other words, we introduce a small amount of Bias into how the New Line is fit to the data. But in return for that small amount of Bais, we get a significant drop in Variance. So, by starting with a slightly worst fit, Ridge Regression can provide better long term predictions.

- Doing **Ridge Regression** is because small sample size can lead to poor Least Squares estimates that result in terrible machine learning predictions.
- **Ridge Regression** helps reduce Variance by shrinking parameters and making predictions less sensitive to them.
- When the sample size are relatively small, the **Ridge Regression** can improve redictions made from new data(i.e. reduce Variance) by making the predictions less sensitive to the Training Data. This is done by adding the **Ridge Regression Penalty** to the thing that must be minimized.
- The **Ridge Regression Penalty** is lambda times the sum of all squared parameters, except for the y-intercept.

#### How Ridge Regression works use continuous variable:
when **Least Squares** determins values for the parameters inthis equation **Size = y-axis intercept + slop * x-axis intercept** it minimizes the sum of the squared residuals. In contrast, when **Ridge Regression** determins values for the parameters inthis equation **Size = y-axis intercept + slop * x-axis intercept** it minimizes the sum of the squared residuals **+ lambda * the slope^2** (slope^2 adds a penalty to the traditional Least Squares method and lambda determines how severe that penalty is). As lambda get larger our predictions for y become less sensitive to x. 
**To decide what value to give lambda,** just try a bunch of values and use Cross Validation typically 10-fold Cross Validation. 

#### you can apply Ridge Regression to discrete variable:

#### you can apply Ridge Regression to Logistic Regression:


### Lasso Regression idea:
- It is very similar to **Ridge Regression**, but it has some very important differences.
- The big diffrence between Ridge and Lasso Regression is that Ridge Regression can only shrink the slope asymptotically close to 0 while Lasso Regression can shrink the slope all the way to 0.
- Lasso Regression can exclude useless variables from equations, so it is a little better than Ridge Regression at reducing the Variance in models that contain a lot of useless variables.
