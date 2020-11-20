# Test 3

Leslie Le and Joeseph Salerno

## Feature Transformation
For feature selection, the data was read in as a DataFrame. Then, we binarized the target feature, from RB and NRB to 1 and 0, respectively. Next, we replaced all the cells in the csv file that was used to represent NaN values to numpy's NaN value. Then, instead of removing those values, we tried to replace the missing values with the mean or the median. There are two separate image files in the repository because using the mean or the median yielded a selection of different features. In the end we decided to replace the missing values using the mean of the features because all the data points are relatively cloe together in terms of values.

### Feature Selection
We used a **Random Forest** for feature selection. Using the Random Forest, the algorithm determined all features that had the most importance to predicting the target feature. Because we did not explicitly define the threshold parameter for the selection process, the algorithm automatically set the threshold at 0.024, which cut the number of features down to 20. Based on the results of the model, we will add or remove features that will help improve the model's predictions. The top 5 most important features, according to the Random Forest are SpMax_B, SpMax_L, SM6_B, SpPosA_B, and SpMax_A.
The **heatmap correlation values** was also used. Using the top twenty feature determine from the random forest, we put the twenty features and the target into a heatmap to determine their correlation to one another. According to the heat map, of the 20 feature from the random forest, the top 5 correlated feature to the target feature are SpMax_L, Sp_Max_A, SM6_B, SpPosA_B, and HyWi_B.
You can see that the feature that are the most important to determining the features, that both models agree on, are SpMax_L, SpMax_A, SpMax_B, SM6_B, and SpPosA_B.
We then removed features from the bottom 10 of these top 20 correlated features. We removed them based on how they impacted the metrics when removed and how correlated they were to the rest of the features. In the end, we removed C%, F03CO, J_Dz, HyWi_B, because they improved the performance of the model when removed (by improved we mean they caused a net increase in the metrics when removed).


### Feature Extraction: PCA (KPCA)
We used the **Principal Component Analysis (PCA)** as the feature extraction method. We tested two separate classifiers, one being regular PCA and the other being KPCA (Kernel). The result we got from them did not differ at all, which implies that we do not need a nonlinear transformation of the data for the model. For PCA and the KPCA, our output consisted of a graph displaying the separation of the classes. In this graph, it is evident that the classes are laid on top of one another, demonstrating that these classes are not linearly separable. The use of both methods of PCA gave us a lead on what classifier might perform best on the dataset because it is not linear and we wouldn't need a specific transformation.

### Feature Regularization
We regularized the data using L1 and L2 regularization.
Looking at L1 Regularization, we can see that when we don't scale the data, having a C value of 0.01 has an increasing Cross validation score. Similarly with a scaling of "1/n_samples". Having a C value of 10, the model still has room to improve the CV score.
With L2 Regularization, with no scaling of the data, a C value of 0.01 appears to have room for improvement, however the graph cuts off at that points, so any further inferences of the graph would be extrapolation. When scaling the data using "1/n_samples", we can see that a C value of 1 could further improve the CV scores, however it seems as if the scores are slowly stagnating. So, it seems best to use a C score in the range of 10 to 25 when creating the model based on the visuals.

## Model: SVM
For this model, we decided to use the SVC classifier api call from Sklearn for convenience purposes and efficiency. Next, we decided to do a grid search to figure out the best hyperparameters for our model, basing the grid search off accuracy because that seemed to be our best performing metric. The resulting kernel being rbf and the C value being 20. I tested the other kernels as well, as mentioned in scaling, and tried a couple of other C values and it appears that any C value between 20 and 25 seems to work best. This is because outside of this range accuracy and the other metrics begin to fall, but within the range Precision and Recall both fall and rise in relation to each other. In the end, we decided to choose a C value of 22 because that was the case where Recall was the highest and closest to being over 0.9 (F1 and accuracy also being at their peak).  Finally, we also managed the random_state of our train_test split for both our regular train/test and our validation. To find the best possible states we used a for loop that checked each value for both and ended up with random_states of 134 and 54. Now, these are not essential to the model, but they allow the model to frequently produce its best performing state as opposed to being randomized. 

## Scaling
For scaling we tested as many scalars as we could on the given dataset. We checked MinMax, Robust, MaxAbs, QuantileTransformer, PowerTransformer and Standard. Every one of these performed poorly besides Robust and Standard, and by performed poorly we mean they significantly lowered the scores of the 4 metrics we tested. In terms of Robust and Standard, Standard performed significantly better on each of the 4 metrics than Robust by a margin 0f 0.6 in some of the metrics. Thus, we decided to use the StandardScaler on our dataset. Now the reasoning of this is because the rbf kernel, the kernel we selected for the hyperparameter of our SVC classifier, already assumes that the data being inputted is centered around 0 (as does L1 and L2 regularization). So, other scalars might completely mess up the rbf kernel’s ordering of magnitude by not being around 0. Now, we did test other scalars with other kernels just to make sure this scalar was the best. We tested each scalar on each kernel and in the end, the combination of rbf and Standard scaling performed the best, with none of the other options breaking 0.9 accuracy. 


## Why would this model be generalizable?

## Sources
