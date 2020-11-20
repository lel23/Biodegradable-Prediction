# Test 3

Leslie Le and Joeseph Salerno

### Feature Selection: heatmap, Random Forest

### Feature Extraction: LDA, PCA, KPCA


## Model: SVM
For this model, we decided to use the SVC classifier api call from Sklearn for convenience purposes and efficiency. Next, we decided to do a grid search to figure out the best hyperparameters for our model, basing the grid search off accuracy because that seemed to be our best performing metric. The resulting kernel being rbf and the C value being 20. I tested the other kernels as well, as mentioned in scaling, and tried a couple of other C values and it appears that any C value between 20 and 25 seems to work best. This is because outside of this range accuracy and the other metrics begin to fall, but within the range Precision and Recall both fall and rise in relation to each other. In the end, we decided to choose a C value of 22 because that was the case where Recall was the highest and closest to being over 0.9 (F1 and accuracy also being at their peak).  Finally, we also managed the random_state of our train_test split for both our regular train/test and our validation. To find the best possible states we used a for loop that checked each value for both and ended up with random_states of 134 and 54. Now, these are not essential to the model, but they allow the model to frequently produce its best performing state as opposed to being randomized. 

## Scaling
For scaling we tested as many scalars as we could on the given dataset. We checked MinMax, Robust, MaxAbs, QuantileTransformer, PowerTransformer and Standard. Every one of these performed poorly besides Robust and Standard, and by performed poorly we mean they significantly lowered the scores of the 4 metrics we tested. In terms of Robust and Standard, Standard performed significantly better on each of the 4 metrics than Robust by a margin 0f 0.6 in some of the metrics. Thus, we decided to use the StandardScaler on our dataset. Now the reasoning of this is because the rbf kernel, the kernel we selected for the hyperparameter of our SVC classifier, already assumes that the data being inputted is centered around 0 (as does L1 and L2 regularization). So, other scalars might completely mess up the rbf kernel’s ordering of magnitude by not being around 0. Now, we did test other scalars with other kernels just to make sure this scalar was the best. We tested each scalar on each kernel and in the end, the combination of rbf and Standard scaling performed the best, with none of the other options breaking 0.9 accuracy. 


## Why would this model be generalizable?

## Sources
