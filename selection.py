"""
Feature Selection
Test 3

Random Forest, heatmap
"""

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from mlxtend.plotting import heatmap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
feat_labels = X.columns

## Random Forest Feature Selection ##
stdsc = StandardScaler()
X = stdsc.fit_transform(X)

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("rf_selection.png")
plt.show()

sfm = SelectFromModel(forest, prefit=True)
X_selected = sfm.transform(X)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])
print("Threshold %f" % np.mean(importances))

cols = []
for f in range(X_selected.shape[1]):
    cols.append(feat_labels[indices[f]])    
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


## HEAT MAP using the above features ##
cols.append('TAR')
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()
