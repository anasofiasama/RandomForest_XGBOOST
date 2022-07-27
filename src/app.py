# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Importación de datos
url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, index_col=[0,3])

# Se elimina la variable Cabin que no aporta para el análisis
df=df.drop(columns='Cabin')

# Se completan los NA con la media en el caso de Age y con la moda en Embarked
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# Se transforma el tipo de variable a categórica cuando corresponde
df[['Sex','Embarked','Survived']]=df[['Sex','Embarked','Survived']].astype('category')

# Se definen las variables explicarivas y objetivo
y=df['Survived']
X=df.drop(columns=['Ticket','Survived']).copy()

# Encoding
X['Sex']=X['Sex'].cat.codes
X['Embarked']=X['Embarked'].cat.codes

# Se separan la muestra de entrenamiento y prueba
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1107)

## RANDOM FOREST
# Modelo que mejor clasifica luego de probar distintos hiperparámetros
best_clas=RandomForestClassifier(n_estimators= 1550,min_samples_split= 2,min_samples_leaf= 2,max_depth= 10,criterion= 'gini',
 bootstrap= True,class_weight=None)

best_clas.fit(X_train,y_train)
y_pred_best=best_clas.predict(X_test)

import pickle
filename = '/workspace/RandomForest/models/finalized_model.sav'
pickle.dump(best_clas, open(filename, 'wb'))

## BOOSTING
# Mejores parámetros luego de la búsqueda grid.
best_params={'colsample_bytree': 0.7, 'eta': 0.05, 'gamma': 0.4, 'max_depth': 3, 'min_child_weight': 1, 'objective': 'binary:hinge'}

best_clf_xgb=xgb.XGBClassifier(**best_params)
best_clf_xgb.fit(X_train,y_train)
best_xgb_pred=best_clf_xgb.predict(X_test)

filename = '/workspace/RandomForest/models/finalized_model_XGB.sav'
pickle.dump(best_clas, open(filename, 'wb'))