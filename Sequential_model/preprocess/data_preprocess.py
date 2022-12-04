# %%
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import random
import os

# %%
def split_by_sub(df, subs):
    df_by_sub = []
    for s in subs:
        df_by_sub.append(df[df.Subject == s])

    return df_by_sub

# %%
def group_by_labels(df_by_subs):
    l0 = []
    l1 = []
    for single_df in df_by_subs:

        if single_df.label.iloc[0] == 0:
            l0.append(single_df)
        if single_df.label.iloc[0] == 1:
            l1.append(single_df)
    return l0, l1

# %%
def create_folds(df, subs_0, subs_1):

    f1 = df[df.Subject.isin(subs_1[:12] + subs_0[:1])].reset_index().drop(columns = 'index')
    f2 = df[df.Subject.isin(subs_1[12:24] + subs_0[1:2])].reset_index().drop(columns = 'index')
    f3 = df[df.Subject.isin(subs_1[24:36] + subs_0[2:3])].reset_index().drop(columns = 'index')
    f4 = df[df.Subject.isin(subs_1[36:48] + subs_0[3:4])].reset_index().drop(columns = 'index')
    f5 = df[df.Subject.isin(subs_1[48:] + subs_0[4:])].reset_index().drop(columns = 'index')
    
    return [f1, f2, f3, f4, f5]

# %%
def upsampled_folds(folds):
    # resampled_folds = []
    data = pd.concat(folds, axis = 0, ignore_index = True)
    
    cols = data.columns.tolist()[1:]
    
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    
    smote = SMOTE()
    resampled_X, resampled_y = smote.fit_resample(X, y)

    f = pd.DataFrame(pd.concat([resampled_X, resampled_y], axis = 1, ignore_index = True))
    f.columns = cols
    

    # X, y = fold.iloc[:, 1:-1], fold.iloc[:, -1]
    # smote = SMOTETomek(random_state = 42)
    # X_resampled, y_resampled = smote.fit_resample(X, y)
    # f = pd.DataFrame(pd.concat([X_resampled, y_resampled], axis = 1, ignore_index = True))
    # resampled_folds.append(f)

    return f

# %%
df = pd.read_csv('../dataset/chairrise_combined_3_trials.csv').drop(columns = 'Unnamed: 0')
df
# %%
cols = df.columns.tolist()

numeric_cols = cols[1:-1]

scaler = StandardScaler().fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])

df
# %%

# categorical_col = ['sex']
# encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore').fit(df[categorical_col])
# encoded_cols = list(encoder.get_feature_names(categorical_col))

# df[encoded_cols] = encoder.transform(df[categorical_col])

# %%
# df = df[df.columns.tolist()[:-4] + encoded_cols + ['label']]
# df
# %%
df.label.unique()
# %%
l0_subs = df[df.label == 0].Subject.unique().tolist()
l1_subs = df[df.label == 1].Subject.unique().tolist()
# l2_subs = df[df.label == 2].Subject.unique().tolist()
l0_subs

len(l0_subs), len(l1_subs)
# %%
# len(n1)
# # %%
# sum(n1[:35]), sum(n0[:35])

# %%
folds = create_folds(df, l0_subs, l1_subs)
folds[0]

# %%
random.shuffle(folds)
folds[0]
# %%
from collections import Counter
for fold in folds:
    print(Counter(fold.label))
# %%

train_set = upsampled_folds([folds[0], folds[1], folds[3]])


train_set

# %%
Counter(train_set.label)
# %%
val_set = folds[4].drop(columns = 'Subject')
test_set = folds[2]

# %%
train_set.to_csv('../dataset/chairrise_train.csv')
val_set.to_csv('../dataset/chairrise_val.csv')
test_set.to_csv('../dataset/chairrise_test.csv')
# %%
val_set
# %%
from collections import Counter

Counter(val_set.label)
# %%

for idx, input_seq in train_set.iterrows():
    print(input_seq.label)
    break
# %%
'BW1' in test_set.columns.tolist()
# %%
import pandas as pd
df = pd.read_csv('../dataset/chairrise_train.csv').drop(columns = 'Unnamed: 0')
df

# %%
df.sample(frac = 1)
df

# %%
df.to_csv('../dataset/chairrise_train.csv')
# %%
count = 0
for col in train_set.columns.tolist():
    if col.endswith('__2'):
        break

    count += 1

print(count)
# %%
train_set.columns.tolist()
# %%
