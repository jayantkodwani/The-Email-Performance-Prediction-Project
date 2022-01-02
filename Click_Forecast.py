#Email_Forecast

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

email=pd.read_csv("C:/Users/v-jakodw/Downloads/Email Forecast Model/Emails.csv",header=0,index_col=None)
#email=pd.read_csv("https://github.com/jayantkodwani/MarketoEmailPerformance/blob/main/Emails.csv",header=0,index_col=None)

email.drop(['Fiscal Year', 'Fiscal Quarter','Fiscal Month','Email Program Name','Delivered%','CTR%','Open Rate%','Row#'], axis = 1, inplace = True)
print (email.head())

email['Opens'] = pd.to_numeric(email['Opens'].str.replace(",", ""), errors='coerce')
email['Clicks'] = pd.to_numeric(email['Clicks'].str.replace(",", ""), errors='coerce')
print (email.head())

X = email[['Opens','Play Num','Solution Area Num']]
y = email['Clicks'].apply(lambda x: float(x))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Train the model

lm = LinearRegression() 
lm.fit(X_train,y_train)

y=lm.predict([[1000,5,1]])[0]
print(y)

# save the model to disk
filename = 'email_click_model.sav'
pickle.dump(lm, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
