"""
Test 3
SVM
"""

from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer, QuantileTransformer, PowerTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    classifier.fit(X, y)
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

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


#Remove features that cause a net increase in metrics when removed and
#target feature, obv

 
X = df[[i for i in list(df.columns) if i != 'TAR' and i!= 'C%'
        and i!= 'F03CO' and i!= 'J_Dz'and i!= 'HyWi_B' and i!= ''
        ]]



y = df['TAR']
feat_labels = X.columns


#5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state=131)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25,
                                                  random_state=2)


# Standardizing the features:
    #Scalars: MinMax - lowers all scores
    #         Robust - performs slightly worse than standard
    #         MaxAbs - Lowers all scores
    #         QuantileTransformer - Lowers all scores
    #         powerTransformer - Lowers all scores
     
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#random forest feature selection
forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
sfm = SelectFromModel(forest, prefit=True)
X_selected = sfm.transform(X)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])
print("Threshold %f" % np.mean(importances))
# Now, let's print the  features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):
cols = []
for f in range(X_selected.shape[1]):
    cols.append(feat_labels[indices[f]])    
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
X_train_std = X_train_std[:, :X_selected.shape[1]]
X_test_std = X_test_std[: , :X_selected.shape[1]]

##
'''
param_grid = {'C': [10, 15, 20, 25, 30, 35, 40, 45, 50], 
              'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
               'random_state': range(0,30),
               'gamma': ['scale', 'auto']}

lg = GridSearchCV(SVC(), param_grid, verbose = 0, scoring = 'accuracy')
lg.fit(X_train_std, y_train)
print()
print(lg.best_params_)
'''

svm = SVC(kernel='rbf', C=20.0, random_state=0, gamma = 'auto')


scores = cross_val_score(svm, X_train, y_train, cv=5)
print("CV Train Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(svm, X_test, y_test, cv=5)
print("CV Test Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(svm, X_val, y_val, cv=5)
print("CV Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

svm.fit(X_train_std, y_train)
svc_pred = svm.predict(X_test_std)


X_combined_std = np.vstack((X_train_std[:, 1:], X_test_std[:, 1:]))
y_combined = np.hstack((y_train, y_test))
#plot_decision_regions(X=X_combined_std, y=y_combined, classifier=SVC(kernel='rbf', C=10.0, random_state=1))
#plt.savefig("svm.png")
#plt.show()

print(classification_report(y_test, svc_pred))

print("SVM Testing Accuracy: %.3f" % accuracy_score(y_test, svc_pred))
print("SVM Testing F1-Score: %.3f" % f1_score(y_test, svc_pred))
print("SVM Testing Precision: %.3f" % precision_score(y_test, svc_pred))
print("SVM Testing Recall: %.3f" % recall_score(y_test, svc_pred))


