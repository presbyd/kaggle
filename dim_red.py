# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from kaggle.api.kaggle_api_extended import KaggleApi
import tita_functions as tf

# %%
test_data = pd.read_csv("./test.csv")
train_data = pd.read_csv("./train.csv")

# %%
train_data.head(5)
test_data.head(5)


# %%
# merge test and train data
merged = pd.concat([train_data, test_data], sort=False).reset_index(drop=True)
merged.head()

# %%
merged_num_f = merged.select_dtypes(np.number)
merged_num_f = merged_num_f.dropna()
y_num = merged_num_f['Survived']
merged_num = merged_num_f.drop(columns=["Survived"], axis=1)

# %%
stdScaler = StandardScaler()
stdScaler.fit(merged_num)
xTrainScaled = stdScaler.transform(merged_num)


pca = PCA(n_components=3)  # Projection to 2d from 47d
pca.fit(xTrainScaled)
pcaTrain = pca.transform(xTrainScaled)
trainPca = pd.DataFrame(data=pcaTrain, columns=["pca-1", "pca-2", "pca-3"])
print("Projection to 2D from 47D:")
trainPca.head()
finalDf = pd.concat([trainPca, y_num], axis=1)

# %%
fig, ax = plt.subplots(1, 1, figsize=(18, 7))
ax.set_xlabel("PCA_1", fontsize=15)
ax.set_ylabel("PCA_2", fontsize=15)
ax.set_title("2-Component PCA (2D-Transformed Samples)", fontsize=20)
targets = [1, 0]
colors = ["g", "r"]
for target, color in zip(targets, colors):
    indices = finalDf["Survived"] == target
    ax.scatter(finalDf.loc[indices, "pca-1"],
               finalDf.loc[indices, "pca-2"],
               c=color, s=37)
plt.legend(targets)
plt.show()

# %%
sns.scatterplot(x=finalDf['pca-1'], y=finalDf['pca-2'],
                hue=finalDf['Survived'])
