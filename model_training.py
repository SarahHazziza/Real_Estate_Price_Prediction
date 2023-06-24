from madlan_data_prep import prepare_data
import pandas as pd
import re
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn import linear_model
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error


data =  pd.read_excel('output_all_students_Train_v10.xlsx')

df = prepare_data(data)

missing_values = [col for col in df.columns if df[col].isnull().any()]

# Printing the number of missing values and percentage of missing values in each column
for col in missing_values:
    print(col, round(df[col].isnull().mean(), 3), ' % missing values')

#Dropping columns that have more than 15% missing values
try:
    df.drop(['total_floors', 'number_in_street', 'publishedDays '], axis=1, inplace=True)
except:
    pass

#Dropping columns that could lead to multicolinearity, since we created new columns thanks to them or since they are too complex (e.g. description)
try:
    df.drop(['floor_out_of', 'description '], axis=1, inplace=True)
except:
    pass

#Creating list of numerical columns and categorical columns
num_cols = [col for col in df.columns if df[col].dtypes!='O']
cat_cols = [col for col in df.columns if (df[col].dtypes=='O')]

df_selected = df[cat_cols + num_cols]

# Label encoding for categorical columns to be able to analyze the data and perform feature engineering
encoder = LabelEncoder()
df_encoded = df_selected.copy()
df_encoded[cat_cols] = df_selected[cat_cols].apply(lambda x: encoder.fit_transform(x))

#Visualizing outliers 
for col in df_encoded.corr()[df_encoded.corr()['price'] > 0.3].index:
    if col == 'price':
        pass
    else:
        sns.regplot(x=df_encoded[col], y=df_encoded['price'])
        plt.show()

corr_matrix = df_encoded.corr() #creating correlation matrix

# Generate a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

plt.figure(figsize=(10, 15))
# Plotting the heatmap of correlation of the features with the target variable 'price'
sns.heatmap(df[num_cols].corr()[['price']].sort_values(by='price', ascending=False), annot=True, cmap='viridis')

#Removing outliers in high correlation with the target column
df['room_number'] = df['room_number'].replace(35,np.NaN) 
df['Area'] = df['Area'].replace(1000,np.NaN)

#Filling missing values in high correlation with target column with seems to be the best option
df['area_per_room'] = df['Area']/df['room_number'] #creating new column of area per room to get average square meter per room 
df['Area'] = df['Area'].fillna(df['room_number'] * df['area_per_room'].mean()) #filling missing values thanks to number of rooms and average square meter per room
df.drop('area_per_room',axis=1, inplace = True) #removing help column

#Standardize type column by unifying type of properties that are approximatly the same
df['type'] = df['type'].replace(['דירת גג', 'מיני פנטהאוז'],'פנטהאוז').replace(['דופלקס', 'טריפלקס', 'דירת נופש'],'דירה').replace(['קוטג טורי', 'קוטג', 'דו משפחתי', 'בניין'],'בית פרטי').replace(['נחלה','מגרש'],'אחר')

#Using information gain filtering method to calculate the reduction in entropy from the transformation of the data. Helps us understand the importance of each feature for the target. 
X_analyze = df_encoded.drop('price', axis=1) 
y_analyze = df['price']

imputer = SimpleImputer(strategy='median')
X_analyze = imputer.fit_transform(X_analyze)

importances = mutual_info_classif(X_analyze, y_analyze)
feat_importances = pd.Series(importances, df_encoded.columns[0:len(df_encoded.columns)-1])
feat_importances.plot(kind='barh', color='teal')
plt.show()

#Removing column room_number because of its high correlation with Area even before filling Area's missing values(see their correlation coefficient), that could lead to multicollinearity
try:
    df.drop(['room_number'],axis=1, inplace = True)
except:
    pass

#Removing columns with relatively low importance values, indicating low impact on the target price
try:
    df.drop(['handicapFriendly ', 'hasParking ', 'hasMamad ', 'hasBars '],axis=1, inplace = True)
except:
    pass

#Assigning relevant columns to X and y to perform model
X = df.drop('price', axis=1)
y = df.price

#Creating new list of numerical columns and categorical columns
num_cols_model = [col for col in X.columns if X[col].dtypes!='O']
cat_cols_model = [col for col in X.columns if (X[col].dtypes=='O')]

#Creating pipelines for the model
numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

column_transformer = ColumnTransformer([
     ('numerical_preprocessing', numerical_pipeline, num_cols_model),
    ('categorical_preprocessing', categorical_pipeline, cat_cols_model)
    ], remainder='drop')

pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', linear_model.ElasticNet(alpha=0.1, l1_ratio=0.99))
])

pipe_preprocessing_model.fit(X, y)

#Adding trained model to pkl file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(pipe_preprocessing_model, file)

#Performing 10 fold cross-validation
cv_scores = cross_val_score(pipe_preprocessing_model, X, y, cv=10, scoring='neg_mean_squared_error')

cv_scores = -cv_scores

#Checking performance of our model with RMSE, R-squared and MAE
RMSE = np.sqrt(cv_scores.mean())
R_squared = np.mean(cross_val_score(pipe_preprocessing_model, X, y, cv=10, scoring='r2'))

print(f"10-fold Cross-Validation Results:")
print(f"RMSE: {np.round(RMSE, 2)}, R-Squared: {np.round(R_squared, 2)}")

#We will now try our model with the pkl file on a sample data made for testing
test_data = pd.read_excel('Dataset_for_test.xlsx')
test_data = prepare_data(test_data)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

X_test = test_data.drop('price', axis=1) 
y_test = test_data['price']

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
