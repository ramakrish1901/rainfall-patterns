import numpy as np
import pandas as pd
import matplotlib . pyplot as plt
import seaborn as snsfrom sklearn . utils import resample
from sklearn . model selection import train
t e s t split , GridSearchCV , cross val score
from sklearn . ensemble import RandomForestClassifier
from sklearn . metrics import classification
import pickle
# laod the dataset to a pandas dataframe
data = pd. read csv (”data / Rainfall . csv”)
print ( type ( data ) )
data [”day” ]. unique ()
print (”Data Info :”)
data . info ()
# remove extra spaces in all columns
data . columns = data . columns . str . strip ()
data = data . drop(columns=[”day” ])
# checking the number of missing values
print ( data . isnull () .sum() )
# handle missing values
r e port , confusion matrix , accuracy score
data [” winddirection ”] = data [”winddirection” ]. fillna ( data [”winddirection” ].mode() [0])
data [”windspeed”] = data [”windspeed” ]. fillna ( data [”windspeed” ]. median () )
# checking the number of missing values
print ( data . isnull () .sum() )
# converting the yes & no to 1 and 0 respectively
data [” rainfall ”] = data [” rainfall ” ].map({”yes”: 1, ”no”: 0})
plt . figure ( figsize =(15 , 10) )
for i , column in enumerate ([ ’pressure ’ , ’maxtemp’ , ’temparature ’ , ’mintemp’ , ’dewpoint ’ , ’humidity ’ ,
 ’ cloud ’ , ’sunshine ’ , ’windspeed ’] , 1) :
 plt . subplot (3 , 3, i )
 sns . histplot ( data [column ] , kde=True)
 plt . t i t l e ( f” Distribution of {column}”)
 plt . tight layout ()
plt . show()
plt . figure ( figsize =(6 , 4) )
sns . countplot (x=” rainfall ” , data=data )
plt . t i t l e (” Distribution of Rainfall ”)
plt . show()
# correlation matrix
plt . figure ( figsize =(10 , 8) )
sns . heatmap( data . corr () , annot=True , cmap=”coolwarm” , fmt=”.2 f”)
plt . t i t l e (” Correlation heatmap”)
plt . show()
plt . figure ( figsize =(15 , 10) )
for i , column in enumerate ([ ’pressure ’ , ’maxtemp’ , ’temparature ’ , ’mintemp’ , ’dewpoint ’ , ’humidity ’ ,
 ’ cloud ’ , ’sunshine ’ , ’windspeed ’] , 1) :
 plt . subplot (3 , 3, i )
 sns . boxplot ( data [column ])
 plt . t i t l e ( f”Boxplot of {column}”)
plt . tight layout ()
plt . show()
# drop highly correlated column
data = data . drop(columns=[ ’maxtemp’ , ’temparature ’ , ’mintemp ’ ])
data . to csv ( ’ processed rainfall data . csv ’)
# separate majority and minority class
df majority = data[ data [” rainfall ”] == 1]
df minority = data [ data [” rainfall ”] == 0]
print ( df majority . shape )
print ( df minority . shape )
# downsample majority class to match minority count
df majority downsampled = resample( df majority , replace=False , n samples=len ( df minority ) ,
 r andom state=42)
# shuffle the final dataframe
df downsampled = df downsampled . sample( frac =1, random state=42) . reset index (drop=True)
# split features and target as X and y
X = df downsampled . drop(columns=[” rainfall ”])
y = df downsampled[” rainfall ”]
# splitting the data into training data and test data
 X
t rain , X test , y train , y
t e st = train
t e s t split (X, y, test size =0.2, random state=42)
rf model = RandomForestClassifier ( random state=42)
 param
”n
 grid rf = {
 estimators ” : [50 , 100, 200],
 ” max features”: [”sqrt” , ”log2”] ,
  }
 ” max depth”: [None, 10, 20, 30],
 ” min samples split”: [2 , 5, 10],
 ” min samples leaf”: [1 , 2, 4]
 grid search rf = GridSearchCV(estimator=rf model , paramgrid=param grid rf , cv=5, n jobs=−1, verbose=2)
grid search rf . fit (X train , y train)
best rf model = grid search rf .best estimator
print (”best parameters for Random Forest :”, grid search rf .best params )
cv scores = cross val score(best rf model , X train , y train , cv=5)
print (”Cross−validation scores:”, cv scores)
print (”Mean cross−validation score:”, np.mean(cv scores))
# test set performance
y pred = best rf model .predict (X test)
print (”Test set Accuracy:”, accuracy score(y test , y pred))
print (”Test set Confusion Matrix:\n”, confusion matrix(y test , y pred))
print (”Classification Report:\n”, classification report (y test , y pred))
##Load the saved model and file and use it for prediction
import pickle
import pandas as pd
# load the trained model and feature names from the pickle file
with open(”model.pkl”, ”rb”) as file :
model data = pickle. load( file)
model = model data[”model”]
feature names = model data[”feature names”]
input data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input df = pd.DataFrame([ input data] , columns=feature names)
prediction = best rf model .predict ( input df)
print (”Prediction result :”, ”Rainfall” if prediction[0] == 1 else ”No Rainfall”)
