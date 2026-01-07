import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


tsv1 = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\data\labels.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly
tsv2 = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\data\prostate_stats.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly

patient_data = pd.merge(tsv1, tsv2, left_on=['pseudo_id'], right_on=['pid'], how='inner') # merge both dataframes on pseudo_id and pid
patient_data = patient_data.dropna() # Drop rows with missing values for simplicity 
patient_data = patient_data.drop(columns=['pseudo_id', 'sex', 'pseudo_patid', 'pid', 'cx_px', 'cy_px', 'cz_px', 'cx', 'cy', 'cz', 'set']) # Drop patient_id as it's not a feature for prediction
patient_data = patient_data[patient_data.label != 2] # Remove rows with label 2 as these are not relevant for binary classification
patient_data = patient_data[patient_data.psa != 'NA'] # remove rows with no psa value till i find a better solution
patient_data = patient_data[patient_data.staging != 'primary'] # remove rows with primary staging till i find a better solution
patient_data = patient_data.drop(columns=['staging']) # drop staging column after removing primary values



#p = sns.scatterplot(data=patient_data[patient_data.px == 0], x='age', y='psa', hue='label')
#p = sns.stripplot(data=patient_data[patient_data.px == 1], x='age', y='psa', hue='label', dodge=True, jitter=False)

# Compute the correlation matrix
sns.set_theme(style="white")
data = patient_data
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

'''corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()'''