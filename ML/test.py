# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featurewiz import FeatureWiz
from mrmr import mrmr_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# %%
df_1 = pd.read_csv('dyphi_1.4.csv').rename(columns = {'filename' : 'Subject'})

df_1.Subject = df_1.Subject.apply(lambda x : x.split('_')[0])
subs_1 = df_1.Subject.tolist()
df_1

# %%
len(df_1.columns.tolist())
# %%

df_2 = pd.read_csv('dyphi_PD1.1.csv')
# %%
df_2.filename = df_2.filename.apply(lambda x : x.split('-')[0])
df_2.rename(columns = {'filename' : 'Subject'}, inplace = True)
subs_2 = df_2.Subject.unique().tolist()
subs_2

# %%
len(df_2.columns.tolist())

# %%
df_1.shape[0], df_2.shape[0]
# %%
df = pd.concat([df_1, df_2], axis = 0, ignore_index = True)
df

# %%
from collections import Counter

Counter(df.label)
# %%
df.Subject.unique()
# %%
CG_subs = ['1786', '1847', '1848', '1949', '1950']
# CG_subs = subs_1 + CG_subs

# %%
label_list = []

for sub in df.Subject.tolist():
    if sub in CG_subs:
        label_list.append(0)

    else:
        label_list.append(1)

len(label_list)

# %%
# drop needless features
df.drop(columns = ['sit4_t_1', 'standing5_t', 'last_peak2_t', 'last_peak2_p'], inplace = True)
df
# %%
df.insert(54, 'label', label_list)
df

# %%
df.drop(columns = 'trial', inplace = True)
# %%
df.columns
# %%

# for col in df.columns.tolist()[1:]:
#     plt.title(col)
#     df[df.label == 0][col].plot()
#     break;

cols = df.columns.tolist()[1:-1]
cols
# %%
for col in cols:
    df.drop(df[df[col] < 0].index, inplace = True)

df
# %%
df.columns.tolist()
# %%
features = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None)
X_train_selected = features.fit(df.iloc[:, 1:-1], pd.Series(df.label.astype(int), name='Output'))
selectedFeatures= features.features  ### provides the list of selected features ###
# X, y = df.iloc[:, 1:-1], df.iloc[:, -1]
# selected_features = mrmr_classif(X, y, K = 20)
# %%
selectedFeatures

# %%
df = df[['Subject'] + selectedFeatures + ['label']]
df

# %%
# from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(X_train)

# cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
# d = [n for n in range(len(cumsum))]
# plt.figure(figsize = (10, 10))
# plt.plot(d, cumsum, color = 'red', label = 'cumulative explained variance')
# plt.title('Cumulative Explained Variance as a Function of the Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.xlabel('Principal Components')
# plt.axhline(y = 98, color = 'k', linestyle = '--', label = '98% Explained Variance')
# plt.legend(loc = 'best');

# %%


# %%
from collections import Counter

Counter(df.label)

# %%
import random
subs = df.Subject.unique().tolist()
print(subs)
random.shuffle(subs)

print(subs)

# %%


# %%
# from imblearn.over_sampling import SMOTE
# X, y = feature_selected_df.iloc[:, 1:-1], feature_selected_df.iloc[:, -1]
# smote = SMOTE(random_state = 42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# X_resampled
df
# %%
numeric_cols = df.columns.tolist()[1:-1]
scaler = StandardScaler().fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])

# %%
df

# %%
PD_subs = [x for x in subs if x not in CG_subs]
PD_subs

# %%
len(PD_subs) * .7, len(CG_subs) * .7

# %%

train_df = df[df.Subject.isin(PD_subs[:42] + CG_subs[:3])]
test_df = df[~df.Subject.isin(PD_subs[:42] + CG_subs[:3])]

# %%
from imblearn.over_sampling import SMOTE

X, y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1]
smote = SMOTE(random_state = 42)
X_train, y_train = smote.fit_resample(X, y)

# %%
# X_train.shape

# add Gaussian noise to the data

# CG_train_df = train_df[train_df.Subject.isin(CG_subs)]
# CG_X, CG_y = CG_train_df.iloc[:, 1:-1], CG_train_df.iloc[:, -1]

# # %%
# noise = np.random.normal(0, .001, [CG_X.shape[0], 53])
# noise

# # %%
# X_train = np.concatenate((X_train, CG_X + noise), axis = 0)

# # # %%
# # X_train.shape
# # # %%
# y_train = np.concatenate((y_train, CG_y), axis = 0)
# y_train

# %%
X_test, y_test = test_df.iloc[:, 1:-1], test_df.iloc[:, -1]
 # %%
# from sklearn.decomposition import PCA

# pca = PCA(0.85)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)

# %%
from sklearn.model_selection import GridSearchCV
best_params = dict()

param_grid = {'max_depth' : list(range(2, 20, 2)), 'n_estimators' : [20, 50, 100, 300], 'max_features' : ['sqrt', 'log2', None]}
grid = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, cv = 5, n_jobs = -2)
grid.fit(X_train, y_train)

best_params.update({'RF' : grid.best_params_})

# %%

# %%
param_grid = [
  {'C': [0.1, 1, 10], 'kernel': ['linear']},
  {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 1.0], 'kernel': ['rbf']},
  {'C' : [0.1, 1, 10], 'gamma' : [0.01, 0.1], 'kernel' : ['poly']}
 ]
# gammas = [10, 1.0, 0.1, 0.01]
grid = GridSearchCV(SVC(), param_grid = param_grid, cv = 5, n_jobs = -2)
grid.fit(X_train, y_train)

best_params.update({'SVC' : grid.best_params_})

# %%
param_grid = {'n_neighbors' : list(range(1, 10, 2))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid, cv = 5)
grid.fit(X_train, y_train)

best_params.update({'KNN' : grid.best_params_})

# %%
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'n_estimators': [20, 50, 100, 300]
            # 'num_class': [10]
          }
grid = GridSearchCV(XGBClassifier(), param_grid = param_grid, cv = 5, n_jobs = -2)
grid.fit(X_train, y_train)

best_params.update({'XGB' : grid.best_params_})

# %%
best_params['RF']

# %%
model = RandomForestClassifier(**best_params['RF'], n_jobs = -2)
# model = RandomForestClassifier(max_depth = 2, n_estimators = 100)
model.fit(X_train, y_train)
rf_preds = model.predict(X_test)
accuracy_score(y_test, rf_preds)

# %%
model = KNeighborsClassifier(**best_params['KNN'])
model.fit(X_train, y_train)
knn_preds = model.predict(X_test)
accuracy_score(y_test, knn_preds)

# %%
model = SVC(**best_params['SVC'])
model.fit(X_train, y_train)
svc_preds = model.predict(X_test)
accuracy_score(y_test, svc_preds)

# %%
model = XGBClassifier(**best_params['XGB'], n_jobs = -2)
model.fit(X_train, y_train)
xgb_preds = model.predict(X_test)
accuracy_score(y_test, xgb_preds)

# %%
best_params['RF'], best_params['XGB']
# %%
misclassified_index = list(np.where(knn_preds != y_test))
misclassified_index
# %%
test_df[test_df.label == 0]
# %%
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_test, knn_preds)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

plt.figure(figsize = (10, 6))
plt.title('KNN')
ax = sns.heatmap(cm, annot = True, cmap = 'binary', fmt = '.2%', cbar = False, annot_kws = {'fontsize' : 15})
ax.set_xlabel('\nPrediction')
ax.set_ylabel('Label')

ax.xaxis.set_ticklabels(['CG', 'PD'])
ax.yaxis.set_ticklabels(['CG', 'PD'])
plt.show();

# %%
cm = confusion_matrix(y_test, rf_preds)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

plt.figure(figsize = (10, 6))
plt.title('RF')
ax = sns.heatmap(cm, annot = True, cmap = 'binary', fmt = '.2%', cbar = False, annot_kws = {'fontsize' : 15})
ax.set_xlabel('\nPrediction')
ax.set_ylabel('Label')

ax.xaxis.set_ticklabels(['CG', 'PD'])
ax.yaxis.set_ticklabels(['CG', 'PD'])
plt.show();

# %%
cm = confusion_matrix(y_test, svc_preds)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

plt.figure(figsize = (10, 6))
plt.title('SVC')
ax = sns.heatmap(cm, annot = True, cmap = 'binary', fmt = '.2%', cbar = False, annot_kws = {'fontsize' : 15})
ax.set_xlabel('\nPrediction')
ax.set_ylabel('Label')

ax.xaxis.set_ticklabels(['CG', 'PD'])
ax.yaxis.set_ticklabels(['CG', 'PD'])
plt.show();

# %%
cm = confusion_matrix(y_test, xgb_preds)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

plt.figure(figsize = (10, 6))
plt.title('XGB')
ax = sns.heatmap(cm, annot = True, cmap = 'binary', fmt = '.2%', cbar = False, annot_kws = {'fontsize' : 15})
ax.set_xlabel('\nPrediction')
ax.set_ylabel('Label')

ax.xaxis.set_ticklabels(['CG', 'PD'])
ax.yaxis.set_ticklabels(['CG', 'PD'])
plt.show();
# %%
print(classification_report(y_test, knn_preds))

# %%

# %%
