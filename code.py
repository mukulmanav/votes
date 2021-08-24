#%%
import pandas as pd
# %%
df=pd.read_csv("votes.csv")
# %%
df.isna().sum()
# %%
df
# %%
import numpy as np
# %%
df.replace("?",np.nan,inplace=True)
# %%
df.isna().sum()
# %% to fill the null values
# 1. Subset your column of interest
# 2. Convert it to boolean based on condition
# 3. surround with variablename[..code uptil step 2..]
# %%
republician=df[df["Class Name"]=="republican"]
# %%
democrat=df[df["Class Name"]=="democrat"]
# %%
df.shape
# %%
republician.shape
# %%
democrat.shape
# %%
republician[' handicapped-infants']=republician[' handicapped-infants'].fillna(republician[' handicapped-infants'].mode()[0])
# %%
republician.isna().sum()
# %%
for columnName in republician.columns:
    republician[columnName]=republician[columnName].fillna(republician[columnName].mode()[0])
# %%
for columnName in republician.columns:
    democrat[columnName]=democrat[columnName].fillna(democrat[columnName].mode()[0])

# %%
republician.isna().sum()
# %%
data=pd.concat([republician,democrat],axis=0)
# %%
data
# %%
x=data.drop(["Class Name"],axis=1)
# %%
y=data["Class Name"]
# %%
y.isna().sum()
# %%
x.shape
# %%
type(x)
# %%
x.dtypes
# %%
x=pd.get_dummies(x,columns=x.columns)
# %%
x.dtypes
# %%
type(x)
# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,stratify=y)
x.head()
# %%
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10,oob_score=True)
model.fit(x_train,y_train)
# %%
model.score(x_test,y_test)
# %%
model.oob_score_
# %%
