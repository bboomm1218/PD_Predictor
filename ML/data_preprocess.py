# %%
import os
import numpy as np
import pandas as pd
from featurewiz import FeatureWiz
from sklearn.metrics import accuracy_score

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
# df_1 = remove_outliers(df_1, df_1['total_t'])
# df_2 = remove_outliers(df_2, df_2['total_t'])

# %%
df_1.shape[0], df_2.shape[0]
# %%
df = pd.concat([df_1, df_2], axis = 0, ignore_index = True)
df

# %%
df.Subject.unique()
# %%
CG_subs = ['1786', '1847', '1848', '1949', '1950']
# CG_subs = subs_1 + CG_subs

label_list = []

for sub in df.Subject.tolist():
    if sub in CG_subs:
        label_list.append(0)

    else:
        label_list.append(1)

len(label_list)

# %%
cols_to_drop = ['sit4_t_1', 'standing5_t' ]
df.drop(columns = ['sit4_t_1', 'standing5_t', 'last_peak2_t_1', 'last_peak2_p_1'], inplace = True)
df
# %%
df.insert(54, 'label', label_list)
df

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
new_cols = []
for col in cols[1:]:
    segments = col.split('_')
    # print(segments[-1])
    if len(segments) == 2:
        new_cols.append(f'{segments[0][:-1]}_{segments[1]}_{segments[0][-1]}')
        
        

    elif len(segments) == 4:
        new_cols.append(f'{segments[0]}_{segments[1]}_{segments[2][:-1]}_{segments[3]}_{segments[2][-1]}')
        # new_cols.append(segments[0] + '_' + segments[1] + '_' + segments[2][:-1] + '_' + segments[2][-1])
        
    

    else:
        new_cols.append(f'{segments[0]}_{segments[1][:-1]}_{segments[2]}_{segments[1][-1]}')
        

new_cols
# %%
df
# %%
new_cols = ['Subject', 'total_t'] + new_cols + ['label']

new_cols
# %%
df.columns = new_cols

df
# %%
set_1_cols = ['Subject', 'total_t']
set_2_cols = ['Subject', 'total_t']
set_3_cols = ['Subject', 'total_t']
set_4_cols = ['Subject', 'total_t']

for col in new_cols:
    if col.endswith('_1'):
        set_1_cols.append(col)

    if col.endswith('_2'):
        set_2_cols.append(col)
    
    if col.endswith('_3'):
        set_3_cols.append(col)

    if col.endswith('_4'):
        set_4_cols.append(col)

# %%
set_1_cols.append('label')
set_2_cols.append('label')
set_3_cols.append('label')
set_4_cols.append('label')

# %%

df1 = df.filter(items = set_1_cols)
df2 = df.filter(items = set_2_cols)
df3 = df.filter(items = set_3_cols)
df4 = df.filter(items = set_4_cols)


# %%
df1['trial'] = 1
df2['trial'] = 2
df3['trial'] = 3
df4['trial'] = 4
# %%
cols = df1.columns.tolist()

# %%
cols
# %%
new_cols = []
for col in cols[2:-2]:
    new_cols.append(col[:-2])

new_cols

# %%
new_cols = ['Subject', 'total_t'] + new_cols + ['label', 'trial']
# %%

df1.columns = new_cols
df2.columns = new_cols
df3.columns = new_cols
df4.columns = new_cols

# %%
df1
# %%
df = pd.concat([df1, df2, df3, df4], axis = 0, ignore_index = True)
df
# %%
df.sort_values(by = ['Subject', 'total_t', 'trial'], ignore_index = True, inplace = True)

df
# %%
df.to_csv('./data/chairrise_segmented.csv')

# %%
df = pd.read_csv('./data/chairrise_segmented.csv').drop(columns = 'Unnamed: 0')
df
# %%
cols = df.columns.tolist()[1:-2]
cols

# %%
# %%
subs = df.Subject.unique().tolist()
# %%
df_by_sub = []
for sub in subs:
    temp_df = df[df.Subject == sub]
    for i in range(0, len(temp_df), 4):
        df_by_sub.append(temp_df.iloc[i : i + 4])
    
    del temp_df

# %%        

# %%
df_by_sub[0]

# %%
df_to_concat = []
df_by_sub_reshaped = []
trials_to_combine = []

for df_sub in df_by_sub:

    df_sub = df_sub.reset_index(drop = True)

    for i, row in df_sub.iterrows():   
        if i == df_sub.shape[0] - 2:
            break

        for j in range(3):
            trials_to_combine.append(df_sub.iloc[[i + j]].reset_index(drop = True))

        # print(i)
        df_to_concat.append(pd.concat(trials_to_combine, axis = 1).reset_index(drop = True))
        # del steps_to_combine
        trials_to_combine = []
    # print(df_sub.Subject)
    if len(df_to_concat):

        df_by_sub_reshaped.append(pd.concat(df_to_concat, axis = 0).reset_index(drop = True))

    del df_sub 
    # del step
    # del steps_to_combine
    del df_to_concat 
    df_to_concat = []

# %%
df_by_sub_reshaped[1]
# %%
combined_df = pd.concat(df_by_sub_reshaped, axis = 0, ignore_index = True)
combined_df
# %%
old_cols = combined_df.columns.tolist()
old_cols
# %%
l = len(old_cols) // 3

new_cols = old_cols[:l]
new_cols
# %%
for i in range(2, 4):

    for e in old_cols[:l]:
        new_cols.append(e + f'_{i}')

# %%
print(len(new_cols[:l]))

for i in range(2, 4):
    print(len([x for x in new_cols if x.endswith(f'_{i}')]))

# %%
combined_df.set_axis(new_cols, axis = 1, inplace = True)
combined_df

# %%
for i in range(2, 4):
    combined_df.drop(columns = [f'trial_{i}', f'Subject_{i}', f'label_{i}', f'total_t_{i}'], inplace = True)

combined_df.columns.tolist()
# %%
[x for x in combined_df.columns.tolist() if x.startswith(('total_t', 'Subject', 'trial', 'label'))]
# %%
# combined_df.drop(columns = ['right1_0'], inplace = True)

# %%
combined_df.drop(columns = ['total_t'], inplace = True)
# %%
combined_df.drop(columns = 'trial', inplace = True)
# %%
for x in combined_df.columns.tolist():
    print(x)
# %%
cols = combined_df.columns.tolist()
cols[14]
# %%
len(old_cols)
# %%
cols_ = cols[:14] + cols[15:] + [cols[14]]
cols_
# %%
combined_df = combined_df[cols_]
combined_df

# %%
combined_df.to_csv('./Sequential_model/dataset/chairrise_combined_3_trials.csv')

# %%
