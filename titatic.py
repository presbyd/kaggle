# A walk through thanks to https://www.kaggle.com/code/eraaz1/a-comprehensive-guide-to-titanic-machine-learning
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from kaggle.api.kaggle_api_extended import KaggleApi
import tita_functions as tf

# Plotly visualization
import plotly.graph_objs as go
from plotly.tools import make_subplots
from plotly.offline import iplot, init_notebook_mode


#Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


#Classification (evaluation) metrices
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score


"""Ensembling"""
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import BaggingClassifier
from mlens.ensemble import BlendEnsemble
from vecstack import stacking


#%%
api = KaggleApi()
api.authenticate()

#%%
api.competition_download_file('titanic', "test.csv",  path='./')
api.competition_download_file('titanic', "train.csv",  path='./')
test_data = pd.read_csv("./test.csv")
train_data = pd.read_csv("./train.csv")
#%%
train_data.head(5)
test_data.head(5)


"""For the sake of learning, most of the walkthough is skipped till feature engineering"""
#%%
# merge test and train data
merged = pd.concat([train_data, test_data], sort = False).reset_index(drop=True)
merged.head()

#%%
merged.Cabin.isna().sum()
nanReplaced= merged.Cabin.fillna("X")
merged["cabinProcessed"] = nanReplaced.str.get(0) 

# %%
firstName = merged.Name.str.split(".").str.get(0).str.split(",").str.get(-1)

"""Create a bucket Officer and put Dr, Rev, Col, Major, Capt titles into it."""
firstName.replace(to_replace = ["Dr", "Rev", "Col", "Major", "Capt"], value = "Officer", inplace = True,regex=True)

"""Put Dona, Jonkheer, Countess, Sir, Lady, Don in bucket Aristocrat."""
firstName.replace(to_replace = ["Dona", "Jonkheer", "Countess", "Sir", "Lady", "Don"], value = "Aristocrat", inplace = True,regex=True)

"""Finally Replace Mlle and Ms with Miss. And Mme with Mrs."""
firstName.replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}, inplace = True,regex=True)

"""Replace the Aristocrat with Aristocrat"""
firstName.replace({"the Aristocrat":"Aristocrat"}, inplace = True,regex=True)

"""Insert a column named 'nameProcessed'."""
merged["nameProcessed"] = firstName
plotFrequency(merged.nameProcessed)

#%%

merged["familySize"] = merged.SibSp + merged.Parch + 1
plotFrequency(merged.familySize)

#%%
otherwise = merged.Ticket.str.split(" ").str.get(0).str.get(0) # This extracts the 1st character
merged["ticketProcessed"] = np.where(merged.Ticket.str.isdigit(), "N", otherwise)
plotFrequency(merged.ticketProcessed)

#%%
tf.plotScatterPlot(tf.calculateMissingValues(merged).index,
               tf.calculateMissingValues(merged),
               "Features with Missing Values",
               "Missing Values")
#%%
tf.plotScatterPlot()