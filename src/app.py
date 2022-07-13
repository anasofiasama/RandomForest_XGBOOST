
url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, index_col=[0,3])

df=df.drop(columns='Cabin')

df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df[['Sex','Embarked','Survived']]=df[['Sex','Embarked','Survived']].astype('category')

y=df['Survived']
X=df.drop(columns=['Ticket','Survived']).copy()

X['Sex']=X['Sex'].cat.codes
X['Embarked']=X['Embarked'].cat.codes

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1107)

best_clas=RandomForestClassifier(n_estimators= 1550,min_samples_split= 2,min_samples_leaf= 2,max_depth= 10,criterion= 'gini',
 bootstrap= True,class_weight=None)

 best_clas.fit(X_train,y_train)
 y_pred_best=best_clas.predict(X_test)

 import pickle
filename = '/workspace/RandomForest/models/finalized_model.sav'
pickle.dump(best_clas, open(filename, 'wb'))
