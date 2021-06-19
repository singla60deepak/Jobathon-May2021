#!/usr/bin/env python
# coding: utf-8

# In[4]:


## importing the libraries for data manuplation and preprocessingabs'
import pandas as pd
import numpy as np
import os
import math 
from sklearn.preprocessing import OneHotEncoder


# In[5]:


## import libraries for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns


# In[6]:


## importing libraries for Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid


# In[7]:


## import libraries for performance metrices
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
import sklearn.metrics as metrics


# In[8]:


# checking the current working directory
os.getcwd()


# In[9]:


## reading the training data
df1=pd.read_csv("/Users/deepaksingla/Desktop/company/jobathon/train_s3TEQDk.csv")


# In[10]:


## reading the 30% test data
tdf1=pd.read_csv("/Users/deepaksingla/Desktop/company/jobathon/test_mSzZ8RL.csv")


# In[11]:


## checking shape and head of training data
print(df1.shape)
df1.head()


# In[12]:


## checking shape and head of test data
print(tdf1.shape)
tdf1.head()


# In[13]:


## checking traget variable event rate
print(df1["Is_Lead"].value_counts())
df1["Is_Lead"].value_counts()/df1.shape[0]


# In[14]:


## checking the null values in training data
df1.isnull().sum()


# In[15]:


## checking the null values in test data
tdf1.isnull().sum()


# In[16]:


## checking the dtypes of traing and testing
print(df1.dtypes)
print(tdf1.dtypes)


# In[17]:


## creating a list of var based on their types
catg_var=["Gender","Region_Code","Occupation","Channel_Code","Credit_Product","Is_Active"]
cont_var=["Age","Vintage","Avg_Account_Balance"]
cust_id=["ID"]
target_var=["Is_Lead"]
print(str(catg_var),len(catg_var))
print(str(cont_var),len(cont_var))
print(str(cust_id),len(cust_id))
print(str(target_var),len(target_var))


# In[18]:


## checking the number of unique values for categorivcal var in training data
for i in (catg_var):
    print(i,df1[i].nunique())


# In[19]:


## checking the number of unique values for categorivcal var in test data
for i in (catg_var):
    print(i,tdf1[i].nunique())


# In[20]:


## checking the unique values for categorical var in training data
for i in (catg_var):
    print(i,df1[i].unique())


# In[21]:


## checking the unique values value counts for categorical var in training data
for i in (catg_var):
    print(i,df1[i].value_counts())


# In[22]:


## creating an copy of dataframe
df2=df1.copy()
tdf2=tdf1.copy()


# ## Null value treatment

# In[23]:


## checking the null values in training data
df2.isnull().sum()


# In[24]:


## checking the null values in training data
tdf2.isnull().sum()


# In[25]:


## since thier in null vale in only credit product column, which is categorical we will take null as
## differnt category named as null_value

## doing for training
df2["Credit_Product"]=np.where(df2["Credit_Product"].isna(),"null_val",df2["Credit_Product"])

## doing for test
tdf2["Credit_Product"]=np.where(tdf2["Credit_Product"].isna(),"null_val",tdf2["Credit_Product"])


# In[26]:


df2["Credit_Product"].value_counts()


# In[27]:


print(df2.isnull().sum().sum())
print(tdf2.isnull().sum().sum())
## no null value left


# In[28]:


## descritption of continuous variables
df2[cont_var].describe()


# ## Exploratory Data Analysis (EDA)

# ### Numeric vs. Target EDA

# In[29]:


cont_var


# ### Age vs Target

# In[30]:


fig, ax = plt.subplots()

ax.hist(df2[df2["Is_Lead"]==1]["Age"], bins=15, alpha=0.5, color="blue", label="accept")
ax.hist(df2[df2["Is_Lead"]==0]["Age"], bins=15, alpha=0.5, color="green", label="reject")

ax.set_xlabel("Age")
ax.set_ylabel("Count of customers")

fig.suptitle("Age vs. Is_Lead")

ax.legend();


# In[31]:


fig, ax = plt.subplots()

sns.kdeplot(df2[df2["Is_Lead"]==1]["Age"], shade=True, color="blue", label="accept", ax=ax)
sns.kdeplot(df2[df2["Is_Lead"]==0]["Age"], shade=True, color="green", label="reject", ax=ax)

ax.set_xlabel("Age")
ax.set_ylabel("Density")

fig.suptitle("Age vs. Is_Lead");


# In[32]:


fig, ax = plt.subplots()

sns.boxplot(x="Age", y="Is_Lead", data=df2, orient="h", palette={1:"blue", 0:"green"}, ax=ax)

ax.get_yaxis().set_visible(False)

fig.suptitle("Age vs. Is_Lead")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches);


# ### Vintage vs Target

# In[33]:


fig, ax = plt.subplots()

ax.hist(df2[df2["Is_Lead"]==1]["Vintage"], bins=15, alpha=0.5, color="blue", label="accept")
ax.hist(df2[df2["Is_Lead"]==0]["Vintage"], bins=15, alpha=0.5, color="green", label="reject")

ax.set_xlabel("Vintage")
ax.set_ylabel("Count of customers")

fig.suptitle("Vintage vs. Is_Lead")

ax.legend();


# In[34]:


fig, ax = plt.subplots()

sns.kdeplot(df2[df2["Is_Lead"]==1]["Vintage"], shade=True, color="blue", label="accept", ax=ax)
sns.kdeplot(df2[df2["Is_Lead"]==0]["Vintage"], shade=True, color="green", label="reject", ax=ax)

ax.set_xlabel("Vintage")
ax.set_ylabel("Density")

fig.suptitle("Vintage vs. Is_Lead");


# In[35]:


fig, ax = plt.subplots()

sns.boxplot(x="Vintage", y="Is_Lead", data=df2, orient="h", palette={1:"blue", 0:"green"}, ax=ax)

ax.get_yaxis().set_visible(False)

fig.suptitle("Vintage vs. Is_Lead")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches);


# ### Avg_Account_Balance vs Target

# In[36]:


fig, ax = plt.subplots()

ax.hist(df2[df2["Is_Lead"]==1]["Avg_Account_Balance"], bins=15, alpha=0.5, color="blue", label="accept")
ax.hist(df2[df2["Is_Lead"]==0]["Avg_Account_Balance"], bins=15, alpha=0.5, color="green", label="reject")

ax.set_xlabel("Avg_Account_Balance")
ax.set_ylabel("Count of customers")

fig.suptitle("Avg_Account_Balance vs. Is_Lead")

ax.legend();


# In[37]:


fig, ax = plt.subplots()

sns.kdeplot(df2[df2["Is_Lead"]==1]["Avg_Account_Balance"], shade=True, color="blue", label="accept", ax=ax)
sns.kdeplot(df2[df2["Is_Lead"]==0]["Avg_Account_Balance"], shade=True, color="green", label="reject", ax=ax)

ax.set_xlabel("Avg_Account_Balance")
ax.set_ylabel("Density")

fig.suptitle("Avg_Account_Balance vs. Is_Lead");


# In[38]:


fig, ax = plt.subplots()

sns.boxplot(x="Avg_Account_Balance", y="Is_Lead", data=df2, orient="h", palette={1:"blue", 0:"green"}, ax=ax)

ax.get_yaxis().set_visible(False)

fig.suptitle("Avg_Account_Balance vs. Is_Lead")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches);


# ## Categorical Features vs. Target EDA

# In[39]:


catg_var


# ### Gender

# In[40]:


fig, ax = plt.subplots()
sns.countplot('Gender', hue="Is_Lead", data=df2, 
                palette={1:"blue", 0:"green"}, ax=ax)

plt.close(2) # catplot creates an extra figure we don't need
ax.set_xlabel(i)

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches)
fig.suptitle("Gender vs. Is_Lead");


# In[41]:


#Region_Code
fig, ax = plt.subplots()
sns.countplot('Region_Code', hue="Is_Lead", data=df2, 
                palette={1:"blue", 0:"green"}, ax=ax)

plt.close(2) # catplot creates an extra figure we don't need
ax.set_xlabel("Region_Code")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches)
fig.suptitle("Region_Code vs. Is_Lead");


# In[42]:


#Occupation
fig, ax = plt.subplots()
sns.countplot('Occupation', hue="Is_Lead", data=df2, 
                palette={1:"blue", 0:"green"}, ax=ax)

plt.close(2) # catplot creates an extra figure we don't need
ax.set_xlabel("Occupation")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches)
fig.suptitle("Occupation vs. Is_Lead");


# In[43]:


#Channel_Code
fig, ax = plt.subplots()
sns.countplot('Channel_Code', hue="Is_Lead", data=df2, 
                palette={1:"blue", 0:"green"}, ax=ax)

plt.close(2) # catplot creates an extra figure we don't need
ax.set_xlabel("Channel_Code")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches)
fig.suptitle("Channel_Code vs. Is_Lead");


# In[44]:


#Credit_Product
fig, ax = plt.subplots()
sns.countplot('Credit_Product', hue="Is_Lead", data=df2, 
                palette={1:"blue", 0:"green"}, ax=ax)

plt.close(2) # catplot creates an extra figure we don't need
ax.set_xlabel("Credit_Product")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches)
fig.suptitle("Credit_Product vs. Is_Lead");


# In[45]:


#Is_Active
fig, ax = plt.subplots()
sns.countplot('Is_Active', hue="Is_Lead", data=df2, 
                palette={1:"blue", 0:"green"}, ax=ax)

plt.close(2) # catplot creates an extra figure we don't need
ax.set_xlabel("Is_Active")

color_patches = [
    Patch(facecolor="blue", label="accept"),
    Patch(facecolor="green", label="reject")
]
ax.legend(handles=color_patches)
fig.suptitle("Is_Active vs. Is_Lead");


# In[46]:


## checking correlation of continuous variables
df2[cont_var].corr()
## since max corelation is 0.63 so continuous variables are not much correlated


# ## one hot encoding for categorical variables

# In[47]:


## for training
catg_df=pd.get_dummies(df2[catg_var],prefix=catg_var)

## for test
catg_tdf=pd.get_dummies(tdf2[catg_var],prefix=catg_var)


# In[48]:


## check shape and head of training data
print(catg_df.shape)
catg_df.head()


# In[49]:


## check shape and head of test data
print(catg_tdf.shape)
catg_tdf.head()


# In[50]:


## creating traing data frame after one hot encoding
df3=pd.concat([df2[cust_id], df2[target_var],df2[cont_var],catg_df], axis=1)


# In[51]:


## creating test data frame after one hot encoding
tdf3=pd.concat([tdf2[cust_id],tdf2[cont_var],catg_tdf], axis=1)


# In[52]:


print(df3.shape)
df3.head()


# ## checking correlation

# In[53]:


corr_df=pd.DataFrame(df3.drop(["ID","Is_Lead"],axis=1).corr())
print(corr_df.shape)
corr_df.head()


# In[54]:


## cehcking corr for train
corr=df3.drop(["ID","Is_Lead"],axis=1).corr()
columns=np.full((corr.shape[0],),True, dtype=bool)
for i in range(len(corr)):
    for j in range(len(corr)):
        if corr.iloc[i,j]>0.5 and corr.iloc[i,j]<1.0:
            print(corr.index[i],corr.index[j],corr.iloc[i,j])
            
## max correlation is 0.63 so no need to remove any var


# In[55]:


## checking corr for test
corr_test=tdf3.drop(["ID"],axis=1).corr()
columns=np.full((corr_test.shape[0],),True, dtype=bool)
for i in range(len(corr_test)):
    for j in range(len(corr_test)):
        if corr_test.iloc[i,j]>0.5 and corr_test.iloc[i,j]<1.0:
            print(corr_test.index[i],corr_test.index[j],corr_test.iloc[i,j])
            
## max correlation is 0.62 so no need to remove any var


# ## checking for outliers

# In[56]:


# plotting histogram for continuous var for taring data
for j in (cont_var):
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    bp=ax.boxplot(df3[j])
    plt.title(j)
    plt.show()


# In[57]:


# plotting histogram for continuous var for test data
for j in (cont_var):
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    bp=ax.boxplot(tdf3[j])
    plt.title(j)
    plt.show()


# In[58]:


df3[cont_var].describe()


# In[59]:


## doing outlier treatment for training data
## flooring
for i in (cont_var):
    a=np.percentile(df3[i],25)-1.5*(np.percentile(df3[i],75)-np.percentile(df3[i],25))
    df3[i]=np.where(df3[i]<a,a,df3[i])
    
## capping
for i in (cont_var):
    b=np.percentile(df3[i],75)+1.5*(np.percentile(df3[i],75)-np.percentile(df3[i],25))
    df3[i]=np.where(df3[i]>b,b,df3[i])


# In[60]:


## doing for test data

## flooring
for i in (cont_var):
    a=np.percentile(df3[i],25)-1.5*(np.percentile(df3[i],75)-np.percentile(df3[i],25))
    tdf3[i]=np.where(tdf3[i]<a,a,tdf3[i])
    
## capping
for i in (cont_var):
    b=np.percentile(df3[i],75)+1.5*(np.percentile(df3[i],75)-np.percentile(df3[i],25))
    tdf3[i]=np.where(tdf3[i]>b,b,tdf3[i])


# In[61]:


## splitting the train data into dev(70%) and itv(30%) data


# In[62]:


x_train,x_test,y_train,y_test=train_test_split(df3.drop(target_var,axis=1),df3[target_var],test_size=0.3,random_state=10)


# In[63]:


## Creating and Development data
dev=x_train.copy()
dev[target_var]=y_train
print(dev.shape)
print(dev[target_var].value_counts()/dev.shape[0])
dev.head()


# In[64]:


## Creating and in time validation data
itv=x_test.copy()
itv[target_var]=y_test
print(itv.shape)
print(itv[target_var].value_counts()/itv.shape[0])
itv.head()


# In[65]:


## creating out of time validation data
otv=tdf3.copy()
print(otv.shape)
otv.head()


# In[66]:


## creating a list of variables based on types
# catg_var=["Gender","Region_Code","Occupation","Channel_Code","Credit_Product","Is_Active"]
# cont_var=["Age","Vintage","Avg_Account_Balance"]
cust_id=["ID"]
target_var=["Is_Lead"]
variable_grid=dev.drop(["ID","Is_Lead"],axis=1).columns
print(len(variable_grid))


# ## Building Logistic Regression Model

# In[67]:


# instantiate the model (using the default parameters)
logreg = LogisticRegression()


# In[68]:


# fit the model with data
logreg.fit(dev[variable_grid],dev[target_var])


# In[69]:


# Predicting for dev
y_pred=logreg.predict_proba(dev[variable_grid])

# Predicting for itv
y_itv=logreg.predict_proba(itv[variable_grid])

# Predicting for otv
y_otv=logreg.predict_proba(otv[variable_grid])


# In[70]:


y_pred


# In[71]:


## preparing the score file for dev data
dev_score=dev[cust_id+target_var]
dev_score["prob"]=y_pred[:, 1]
print(dev_score.shape)
print(dev_score.isnull().sum().sum())
dev_score.head()


# In[72]:


## preparing the score file for itv data
itv_score=itv[cust_id+target_var]
itv_score["prob"]=y_itv[:, 1]
print(itv_score.shape)
print(itv_score.isnull().sum().sum())
itv_score.head()


# In[73]:


## preparing the score file for otv data
otv_score=otv[cust_id]
otv_score["prob"]=y_otv[:, 1]
print(otv_score.shape)
print(otv_score.isnull().sum().sum())
otv_score.head()


# In[74]:


## scoring the aucroc
print(roc_auc_score(dev_score["Is_Lead"], dev_score["prob"]))
print(roc_auc_score(itv_score["Is_Lead"], itv_score["prob"]))


# ## Building Random Forest Model

# In[75]:


## defining random forest classifier
rf_clf = RandomForestClassifier(n_estimators=500,criterion='gini',max_depth=4,random_state=0)


# In[76]:


# fit the model with data
rf_clf.fit(dev[variable_grid],dev[target_var])


# In[77]:


# Predicting for dev
y_pred=rf_clf.predict_proba(dev[variable_grid])

# Predicting for itv
y_itv=rf_clf.predict_proba(itv[variable_grid])

# Predicting for otv
y_otv=rf_clf.predict_proba(otv[variable_grid])


# In[78]:


## preparing the score file for dev data
dev_score=dev[cust_id+target_var]
dev_score["prob"]=y_pred[:, 1]
print(dev_score.shape)
print(dev_score.isnull().sum().sum())
dev_score.head()


# In[79]:


## preparing the score file for itv data
itv_score=itv[cust_id+target_var]
itv_score["prob"]=y_itv[:, 1]
print(itv_score.shape)
print(itv_score.isnull().sum().sum())
itv_score.head()


# In[80]:


## preparing the score file for otv data
otv_score=otv[cust_id]
otv_score["prob"]=y_otv[:, 1]
print(otv_score.shape)
print(otv_score.isnull().sum().sum())
otv_score.head()


# In[81]:


## getting aucroc
print(roc_auc_score(dev_score["Is_Lead"], dev_score["prob"]))
print(roc_auc_score(itv_score["Is_Lead"], itv_score["prob"]))


# In[82]:


## preparing the final otv scored file
otv_score=otv_score.reset_index()
otv_score2=otv_score.drop(["index"],axis=1)
otv_score2=otv_score2.rename({"prob":"Is_Lead"},axis=1)
print(otv_score2.shape)
otv_score2.head()


# In[83]:


#otv_score2.to_csv("RF_otv_score_v1.csv",index=False)


# ## Building XGBoost Model

# In[84]:


## creating dataset in dmatrix format for input into xgboost
dtrain_grid=xgb.DMatrix(df3[variable_grid],label=df3[target_var].values)
ddev_grid=xgb.DMatrix(dev[variable_grid],label=dev[target_var].values)
ditv_grid=xgb.DMatrix(itv[variable_grid],label=itv[target_var].values)
dotv_grid=xgb.DMatrix(otv[variable_grid])


# In[85]:


## defining the ratio of non event to evnt
ratio=dev[target_var].value_counts()[0]/dev[target_var].value_counts()[1]
ratio


# In[86]:


## defining Parameter Grid
default_param_grid={
                    'gamma' : [0,1,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'max_depth':[2,3,4],
                    'subsample':[0.6,0.8,1.0],
                    'colsample_by_tree':[0.6,0.8,1.0],
                    'num_boost_round' : [500],
                    'eval_metric' : ['aucpr','logloss'],
                    'scale_pos_weight': [1,np.sqrt(ratio)]
}


# In[87]:


params_df=list(ParameterGrid(default_param_grid))
params_df=pd.DataFrame(params_df)
print(params_df.shape)


# In[88]:


## Defining Evaluation set
eval_set=[(ddev_grid,"Dev"),(ditv_grid,"validation")]


# In[89]:


col_names=["gamma","learning_rate","max_depth","subsample","colsample_by_tree","best_num_trees","scale_pos_weight",'auc_dev',"auc_itv"]
grid_search=pd.DataFrame(columns=col_names)
grid_search


# In[90]:


# ## running the grid serach
# for i in range(len(params_df)):
#     print(i)
    
#     model_grid=xgb.train({
#         'learning_rate':params_df["learning_rate"].iloc[i],
#         'booster':'gbtree',
#         'objective':'binary:logistic',
#         'max_depth':params_df["max_depth"].iloc[i],
#         'gamma':params_df["gamma"].iloc[i],
#         'eval_metric':params_df["eval_metric"].iloc[i],
#         'scale_pos_weight':params_df["scale_pos_weight"].iloc[i],
#         'seed':1,
#         'verbose':False,
#         'subsample':params_df["subsample"].iloc[i],
#         'colsample_by_tree':params_df["colsample_by_tree"].iloc[i],
#         },dtrain=ddev_grid,num_boost_round=params_df["num_boost_round"].iloc[i],
#          early_stopping_rounds=50, evals=eval_set, verbose_eval=False)
    
#     prob_train=model_grid.predict(ddev_grid,ntree_limit=model_grid.best_ntree_limit)
#     auc_dev=roc_auc_score(dev[target_var].iloc[:,0], prob_train)
    
#     prob_itv=model_grid.predict(ditv_grid,ntree_limit=model_grid.best_ntree_limit)
#     auc_itv=roc_auc_score(itv[target_var].iloc[:,0], prob_itv)
    
#     grid_search.loc[len(grid_search)] = [params_df['gamma'].iloc[i],params_df['learning_rate'].iloc[i],
#                                         params_df['max_depth'].iloc[i],params_df['subsample'].iloc[i],
#                                         params_df['colsample_by_tree'].iloc[i],model_grid.best_ntree_limit,
#                                         params_df['scale_pos_weight'].iloc[i],auc_dev,auc_itv]
#     #pd.DataFrame(grid_search).to_csv("grid_search_output_v2.csv")


# In[91]:


# ## checking the grid search output
# grid_search_output_v1=pd.read_csv("grid_search_output_v2.csv")
# print(grid_search_output_v1.shape)
# grid_search_output_v1


# In[92]:


#otv_score2.to_csv("XGB_otv_score_v1.csv",index=False)


# In[93]:


#otv_score2.to_csv("XGB_otv_score_v2.csv",index=False)


# In[94]:


#otv_score2.to_csv("XGB_otv_score_v3.csv",index=False)
#otv_score2.to_csv("XGB_otv_score_v4.csv",index=False)


# ## running another grid search

# In[95]:


## defining Parameter Grid
default_param_grid={
                    'gamma' : [0.1,1,5],
                    'learning_rate':[0.05],
                    'max_depth':[4,5,6],
                    'subsample':[0.8],
                    'num_boost_round' : [500],
                    'eval_metric' : ['aucpr','logloss'],
                    'scale_pos_weight': [1,np.sqrt(ratio)],
                    'seed':[5,10,49]
}

params_df=list(ParameterGrid(default_param_grid))
params_df=pd.DataFrame(params_df)
print(params_df.shape)

col_names=["gamma","learning_rate","max_depth","subsample","eval_metric","seed","best_num_trees","scale_pos_weight",'auc_dev',"auc_itv"]
grid_search=pd.DataFrame(columns=col_names)
grid_search


# In[96]:


# for i in range(len(params_df)):
#     print(i)
    
#     model_grid=xgb.train({
#         'learning_rate':params_df["learning_rate"].iloc[i],
#         'booster':'gbtree',
#         'objective':'binary:logistic',
#         'max_depth':params_df["max_depth"].iloc[i],
#         'gamma':params_df["gamma"].iloc[i],
#         'eval_metric':params_df["eval_metric"].iloc[i],
#         'scale_pos_weight':params_df["scale_pos_weight"].iloc[i],
#         'seed':params_df["seed"].iloc[i],
#         'verbose':False,
#         'subsample':params_df["subsample"].iloc[i],
#         },dtrain=ddev_grid,num_boost_round=params_df["num_boost_round"].iloc[i],
#          early_stopping_rounds=50, evals=eval_set, verbose_eval=False)
    
#     prob_train=model_grid.predict(ddev_grid,ntree_limit=model_grid.best_ntree_limit)
#     auc_dev=roc_auc_score(dev[target_var].iloc[:,0], prob_train)
    
#     prob_itv=model_grid.predict(ditv_grid,ntree_limit=model_grid.best_ntree_limit)
#     auc_itv=roc_auc_score(itv[target_var].iloc[:,0], prob_itv)
    
#     grid_search.loc[len(grid_search)] = [params_df['gamma'].iloc[i],params_df['learning_rate'].iloc[i],
#                                         params_df['max_depth'].iloc[i],params_df['subsample'].iloc[i],
#                                         params_df['eval_metric'].iloc[i],params_df['seed'].iloc[i],
#                                         model_grid.best_ntree_limit,params_df['scale_pos_weight'].iloc[i],
#                                         auc_dev,auc_itv]
#     pd.DataFrame(grid_search).to_csv("grid_search_output_v3.csv")


# In[97]:


#otv_score2.to_csv("XGB_otv_score_v5.csv",index=False)
#otv_score2.to_csv("XGB_otv_score_v6.csv",index=False)


# In[98]:


#otv_score2.to_csv("XGB_otv_score_v7.csv",index=False)


# In[99]:


## based on grid search following is the best hyperparameter value
## Training on 100% of training data
model_v6=xgb.train({'learning_rate':0.05,
                'booster':'gbtree',
                'objective':'binary:logistic',
                'max_depth': 7,
                'seed': 155,
                'colsample_by_tree':0.8,
                #'verbose': False,
                'subsample': 0.8,
#                 'scale_pos_weight':0.9,
                'gamma' : 0.01} , dtrain=dtrain_grid, num_boost_round=238)


# In[100]:


## doing prediction

## for 100% traing data
prob_train=model_v6.predict(dtrain_grid)

## for otv
prob_otv=model_v6.predict(dotv_grid)


# In[101]:


## preparing the score file for dev data
train_score=df3[cust_id+target_var]
train_score["prob"]=prob_train
print(train_score.shape)
print(train_score.isnull().sum().sum())
train_score.head()


# In[102]:


## preparing the score file for otv data
otv_score=otv[cust_id]
otv_score["prob"]=prob_otv
print(otv_score.shape)
print(otv_score.isnull().sum().sum())
otv_score.head()


# In[103]:


## getting aucroc
print(roc_auc_score(train_score["Is_Lead"], train_score["prob"]))


# In[104]:


otv_score=otv_score.reset_index()
otv_score2=otv_score.drop(["index"],axis=1)
otv_score2=otv_score2.rename({"prob":"Is_Lead"},axis=1)
print(otv_score2.shape)
otv_score2.head()


# In[105]:


otv_score2.to_csv("XGB_otv_score_final.csv",index=False)


# ## other performance metrices

# In[107]:


fpr, tpr, threshold = metrics.roc_curve(train_score["Is_Lead"], train_score["prob"])
roc_auc = metrics.auc(fpr, tpr)
roc_auc


# In[108]:


# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[109]:


def KS_table(Y,Y_HAT):
    import pandas as pd
    INPUT = pd.DataFrame({'Y': Y ,'Y_HAT':Y_HAT})
    INPUT['GROUP SCORE'] = pd.qcut(INPUT['Y_HAT'],10,duplicates = 'drop')
   
    KS_TABLE = INPUT.groupby('GROUP SCORE')['Y_HAT'].agg(['min','max','mean','count']).reset_index().sort_values('GROUP SCORE',ascending = False)
    KS_TABLE.columns = ['GROUP SCORE', 'MIN SCORE','MAX SCORE', 'mean_score', 'TOTAL COUNT']
    KS_TABLE['% TOTAL COUNT'] = (100*KS_TABLE['TOTAL COUNT']/sum(KS_TABLE['TOTAL COUNT'])).round(2)
   
    temp = INPUT.groupby('GROUP SCORE')['Y'].agg('sum').reset_index().sort_values('GROUP SCORE', ascending = False)
    temp.columns = ['GROUP SCORE','RESPONDERS']
   
    KS_TABLE = KS_TABLE.merge(temp, on = 'GROUP SCORE' , how = 'inner')
    KS_TABLE['NON RESPONDERS'] = KS_TABLE['TOTAL COUNT'] - KS_TABLE['RESPONDERS']
    KS_TABLE['RESPONSE RATE (%)'] = (100*KS_TABLE['RESPONDERS']/KS_TABLE['TOTAL COUNT']).round(2)
   
    KS_TABLE['CUM TOTAL COUNT'] = KS_TABLE['TOTAL COUNT'].cumsum()
    KS_TABLE['CUM RESPONDERS'] = KS_TABLE['RESPONDERS'].cumsum()
    KS_TABLE['CUM NON RESPONDERS'] = KS_TABLE['NON RESPONDERS'].cumsum()
   
    KS_TABLE['CUM % TOTAL COUNT'] = (100*KS_TABLE['CUM TOTAL COUNT']/sum(KS_TABLE['TOTAL COUNT'])).round(2)
    KS_TABLE['CUM % RESPONDERS'] = (100*KS_TABLE['CUM RESPONDERS']/sum(KS_TABLE['RESPONDERS'])).round(2)
    KS_TABLE['CUM % NON RESPONDERS'] = (100*KS_TABLE['CUM NON RESPONDERS']/sum(KS_TABLE['NON RESPONDERS'])).round(2)
                                       
    KS_TABLE['KS'] = KS_TABLE['CUM % RESPONDERS'] - KS_TABLE['CUM % NON RESPONDERS']
    KS_TABLE['LIFT'] = (1.0*KS_TABLE['CUM % RESPONDERS']/KS_TABLE['CUM % TOTAL COUNT']).round(2)
    KS_TABLE['abs_perc'] = abs(KS_TABLE['mean_score'] - KS_TABLE['RESPONSE RATE (%)']/100)*100
                                       
    ks =  round(max(KS_TABLE['KS']),2)
    decile = np.argmax(KS_TABLE['CUM % RESPONDERS'] - KS_TABLE['CUM % NON RESPONDERS']) + 1
    print('KS =' + str(ks) + " at decile = " + str(decile))
                                       
    diff  = (KS_TABLE['RESPONSE RATE (%)'] - KS_TABLE['RESPONSE RATE (%)'].shift(-1))[:-1]
    breaks = [str(i+1) for i , value in enumerate(diff) if value<0]
    if len(breaks) == 0:
        print(' There are no rank order breaks')
    else:
        print('There are rank order break(s) at decile(s)' + ','.join(breaks))
                                       
    return KS_TABLE


# In[110]:


KS_table(train_score["Is_Lead"], train_score["prob"])


# In[111]:


#######................................Thanks..................................#######

