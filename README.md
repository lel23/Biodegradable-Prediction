

# Biodegradable Prediction
Leslie Le and Joseph Salerno

----

Our goal with this project was to create an accurate generalizable model of materials that were ready and biodegradable versus not ready and biodegradable using different molecular makeups as input. 

We went through a lot of modifying the features to be helpful to the model. Then we inputted those features into a **Support Vector Machine**. We detailed our work in [result.md](https://github.com/lel23/CSCI297-Test3/blob/master/results.md).

### Problems (things that could have been done better)
- Feature Transformation
  - Did you adjust your imputation for features that were not continuous? 

- Feature Selection
  - Compare and analyze feature selection methods: Random Forest with SBS
  - Explain more about the thresholds tests and the number of trees that were used

- Feature Regularization
  - Explain what features were removed when using L1 regularization
  - Testing the feature regularization on different feature sets
  - Specify the scores being used
  - Use more metrics - accuracy, f1 score, precision, etc.
- Model
  - Attempt Linear Discriminant Analysis
  - Go more in-depth about what was done with grid searching
  - Try other types of searching, such as random search
- Random state is not a hyper parameter, as such, it should not be adjusted to attain higher scores in the metrics. We could have used stratified sampling for cross validation, which would help with class imbalance.
