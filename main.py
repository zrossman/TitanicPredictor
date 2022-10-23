from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


api = KaggleApi()
api.authenticate()
api.competition_download_file('titanic', 'train.csv')


#Reading in our dataset
df = pd.read_csv('train.csv')

#Let's take a look at the first row of our dataset to see what we're working with
print(df.iloc[0, :])
print()

#Each row represents a passanger on the titanic, including various information about that passenger, and ultimiately
#whether or not they survived the tragedy.

#Lets first look for any null values in our dataset

print(df.isna().sum())
print()

#So we have 687 NaN values in this column, which means over 77% of our passengers have NaN for 'Cabin'. This is too
#much, so we'll just delete this column, as it will not be useful to us.
df = df.drop(['Cabin'], axis = 1)
print(df.iloc[0, :])
print()

#Let's replace the null values in the age column with the average age for the dataset
average = df['Age'].mean()
df.fillna(average, inplace = True)

#Looking at our data, we also determine that each passenger's name will not be relevant to us, since that really has
#no predictive capability, so we'll go ahead and drop that column. The same goes for the 'Embarked' column, the
#passenger Id column, and the 'Ticket' column.
df = df.drop(['Name', 'Embarked', 'PassengerId', 'Ticket'], axis = 1)
print(df.iloc[0, :])
print()

#At first glance, each of the remaining columns seems like they could be relevant in helping us predect survivors.
#However, we need to change the values in the 'sex' column to numbers, so they can be processed properly.
df['Sex'].replace('female', 0, inplace = True)
df['Sex'].replace('male', 1, inplace = True)
print(df.iloc[0, :])
print()

#Now, lets turn our feature columns and our label column into arrays.
X = np.array(df.drop(['Survived'], axis = 1))
y = np.array(df['Survived'])

#Now we'll split our data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .168, random_state = 42)
print('X_train size:', len(X_train))
print('X_test size:', len(X_test))
print('y_train size:', len(y_train))
print('y_test size:', len(y_test))
print()

#Building our model
rfc = RandomForestClassifier(criterion = 'entropy')
rfc.fit(X_train, y_train)

#Evaluating our model on our training set
rfc_pred_train = rfc.predict(X_train)
print('Training Data Accuracy:', rfc.score(X_train, y_train))
print('Training Data F1 Score:', f1_score(y_train, rfc_pred_train))
train_cm = confusion_matrix(y_train, rfc_pred_train)
print('Training Data Confusion Matrix:')
print(train_cm)
print()

#Evaluating our model on the test set
rfc_pred_test = rfc.predict(X_test)
print('Test Data Accuracy:', rfc.score(X_test, y_test))
print('Test Data F1 Score:', f1_score(y_test, rfc_pred_test))
test_cm = confusion_matrix(y_test, rfc_pred_test)
print('Test Data Confusion Matrix:')
print(test_cm)

#We can see our model is consistently turning out accuracies near 81-82%, shwoing that our decision tree algorithm
#does a sound job of predicting survivors from the Titanic.