import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score


#-----------------------------------------------------------------------------------
# Load Train and Test datasets and create a datasets list
#-----------------------------------------------------------------------------------
house_train = pd.read_csv('C:/Users/invek/Desktop/House_Pricing/train.csv', index_col="Id")
house_test = pd.read_csv('C:/Users/invek/Desktop/House_Pricing/test.csv', index_col="Id")
datas = [house_test, house_train]
#------------------------------------------------------------
# Preprocessing of datasets
#------------------------------------------------------------
cat_col = house_train.select_dtypes(include="object").columns
for data in datas:
    data['LotFrontage'].fillna(value=data['LotArea']**(1/2), inplace=True)
    data['PoolArea'] = data['PoolArea'].apply(lambda x: int(x > 0))
    data[pd.get_dummies(data['MSSubClass'], prefix='m').columns] = pd.get_dummies(data['MSSubClass'], prefix='m')
    data.drop('MSSubClass', inplace=True, axis=1)
    for col in cat_col:
        data[pd.get_dummies(data[col]).columns] = pd.get_dummies(data[col])
    data.drop(cat_col, inplace=True, axis=1)
#-------------------------------------------------------------------------
# Create X dataframe from intersection of train and test datasets columns, because categorical data can be different
#-------------------------------------------------------------------------
int_col = set(house_train.columns.to_list()) & set(house_test.columns.to_list())
X = house_train[int_col]
y = house_train['SalePrice']
#-------------------------------------------------------------------
# Creating Regression XGB model with parameters previsioly founded with FridSearch. Chech model scores with
# cross validation to 5 parts.
#-------------------------------------------------------------------
model = xgb.XGBRegressor(random_state=1, n_estimators=250, reg_lambda=0.8, max_depth=7, learning_rate=0.04)
scores = -1 * cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print("MAE scores:\n", sum(scores)/5)
model.fit(X, y)
#-------------------------------------------------------
# See which features is important to model prediction and how much
#-------------------------------------------------------
# z = zip(X.columns.tolist(), model.feature_importances_)
# for col, fea in z:
#     print(col, ' ', fea*100)
#----------------------------------
# Create prediction of test data and seve it ti csv file
#----------------------------------
# predictions = pd.read_csv("C:/Users/invek/Desktop/House_Pricing/sample_submission.csv")
# predictions.drop("SalePrice", axis=1, inplace=True)
# predictions["SalePrice"] = model.predict(house_test[int_col])
# predictions.to_csv("C:/Users/invek/Desktop/House_Pricing/sample_submission_1.csv", index=False)
