## Code taken from
## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html
# TODO: change x and y to the dataset's features

# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Jaques Grobler <jaques.grobler@inria.fr>
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Reading data
df = pd.read_csv('NewBioDegWCols.csv')
df.columns = ['SpMax_L','J_Dz','nHM','F01','F04','NssssC','nCb-','C%','nCp',
              'n0','F03CN','SdssC','HyWi_B','LOC','SM6_L','F03CO','Me','Mi',
              'nN-N','nArN02','nCRX3','SpPosA_B','nCIR','B01','B03','N-073',
              'SpMax_A','Psi_i_1d','B04','Sd0','TI2_L','nCrt','c-026','F02',
              'nHDon','SpMax_B','Psi_i_A','nN','SM6_B','nArCOOR','nX','TAR']

df['TAR'] = df['TAR'].replace(['RB', 'NRB'], [1, 0])
df.replace(to_replace='NaN', value=np.nan, regex=True, inplace=True)
# df.mean(), df.median()
df.fillna(df.mean(), inplace=True)


X = df[[i for i in list(df.columns) if i != 'TAR']]
y = df['TAR']



rnd = check_random_state(1)

# set up dataset
n_samples = 1056
n_features = len(df.columns)-1

# l1 data (only 5 informative features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# l2 data: non sparse, but less features
y_2 = np.sign(.5 - rnd.rand(n_samples))
X_2 = rnd.randn(n_samples, n_features // 5) + y_2[:, np.newaxis]
X_2 += 5 * rnd.randn(n_samples, n_features // 5)

clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                       tol=1e-3),
             np.logspace(-2.3, -1.3, 10), X_train_std, y_train),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True),
             np.logspace(-4.5, -2, 10), X_train_std, y_train)]

colors = ['navy', 'cyan', 'darkorange', 'blue', 'green']
lw = 2

for clf, cs, X, y in clf_sets:
    # set up the plot for each regressor
    fig, axes = plt.subplots(nrows=2, sharey=True, figsize=(9, 10))

    for k, train_size in enumerate(np.linspace(0.30, 0.70, 4)[::-1]):
        param_grid = dict(C=cs)
        # To get nice curve, we need a large number of iterations to
        # reduce the variance
        grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                            cv=ShuffleSplit(train_size=train_size,
                                            test_size=.3,
                                            n_splits=250, random_state=1))
        grid.fit(X, y)
        scores = grid.cv_results_['mean_test_score']

        scales = [(1, 'No scaling'),
                  ((n_samples * train_size), '1/n_samples'),
                  ]

        for ax, (scaler, name) in zip(axes, scales):
            ax.set_xlabel('C')
            ax.set_ylabel('CV Score')
            grid_cs = cs * float(scaler)  # scale the C's
            ax.semilogx(grid_cs, scores, label="fraction %.2f" %
                        train_size, color=colors[k], lw=lw)
            ax.set_title('scaling=%s, penalty=%s, loss=%s' %
                         (name, clf.penalty, clf.loss))

    plt.legend(loc="best")
plt.show()
