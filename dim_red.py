from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%%
train_data.head(5)
test_data.head(5)


"""For the sake of learning, most of the walkthough is skipped till feature engineering"""
#%%
# merge test and train data
merged = pd.concat([train_data, test_data], sort = False).reset_index(drop=True)
merged.head()