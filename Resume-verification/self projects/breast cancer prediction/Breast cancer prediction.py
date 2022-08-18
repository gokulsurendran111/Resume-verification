#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import seaborn as s
#from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
#optimum parameter choosing 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle
import os 
import warnings
warnings.filterwarnings('ignore')


# In[2]:



data = pd.read_csv('data.csv')
data


# In[3]:


data.shape


# In[4]:


df = data


# In[5]:


df['diagnosis'].value_counts()


# In[6]:


df.dtypes


# In[7]:


df['diagnosis']= df['diagnosis'].astype('category')
df.dtypes


# In[8]:


df.head()


# In[9]:


x= df.drop (labels='diagnosis' ,axis =1 )
x


# In[10]:


x.idxmin()


# In[11]:


x.loc[101,'radius_mean']


# In[12]:


y = df['diagnosis']
y


# In[13]:


x.isnull().sum()


# In[14]:


df_norm = (x- x.mean()) / (x.max()- x.min())
df_norm= pd.concat ([df_norm,y], axis =1 )
df_norm


# In[15]:


df.drop('diagnosis',axis =1).drop('id',axis =1).corr()


# In[16]:


plt.rcParams['figure.figsize']=(20,8)
s.set(font_scale=0.8)
# In co relation 1 is the highest and -1 is lowest
s.heatmap (df.drop('diagnosis',axis =1).drop('id',axis =1).corr(),cmap = 'coolwarm',annot = True)


# In[17]:


x_norm = df_norm.drop (labels= 'diagnosis', axis =1 )
y_norm = df_norm ['diagnosis']
col = x_norm. columns
print (col)
display (x_norm)
display (y_norm)


# In[18]:


le = LabelEncoder()
le.fit (y_norm)
y_norm = le.transform(y_norm)
y_norm = pd.DataFrame(y_norm)
print (y_norm)


# In[19]:


def FitModel (X,Y, algo_name , algorithm, gridSearchParams, cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split (X,Y,test_size = 0.2)
    
    # Find the Parameters , then choose best parameters 

    grid = GridSearchCV(estimator = algorithm, param_grid = gridSearchParams,
                        cv = cv, scoring = 'accuracy', verbose = 1 , n_jobs = -1 )
    
    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict (x_test)
    cm = confusion_matrix (y_test,pred)
    
    print (pred)
    pickle.dump(grid_result,open(algo_name,'wb'))
    
    print ('Best Params :\n', best_params)
    print ('Classification Report:\n',classification_report(y_test,pred))
    print ('Accuracy Score\n', (accuracy_score(y_test,pred)))
    print ('Confusion Matrix :\n',cm)


# In[20]:


#cv = crosss validation 
param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }

FitModel (x_norm,y_norm,'SVC',SVC(), param, cv =5)


# In[21]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (x_norm,y_norm,'Random Forest',RandomForestClassifier(), param, cv =10)


# In[22]:


np.random.seed(10)
x_train,x_test, y_train,y_test = train_test_split (x_norm,y_norm,test_size = 0.2)
forest = RandomForestClassifier (n_estimators = 500)
fit = forest.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Classification Report:\n',classification_report(y_test,predict))
print ('Accuracy Score', (accuracy_score(y_test,predict)))
print ('Accuracy of Random Forest ', (accuracy))
print ('Confusion Matrix :\n',cmatrix)


# In[23]:


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print ("Feature Ranking:")
for f in range (x.shape[1]):
    print ("Feature %s (%f)"  %(list (x)[f],importances[indices[f]]))


# In[24]:


feat_imp = pd.DataFrame({'Feature': list(x), 'Gini importance': importances[indices]})
plt.rcParams['figure.figsize']= (12,12)
s.set_style ('whitegrid')
ax= s.barplot(x ='Gini importance', y = 'Feature', data = feat_imp  )
ax.set (xlabel = 'Gini Importances')
plt.show()



# In[25]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (x_norm,y_norm,'XGBoost', XGBClassifier(),param, cv = 5)


# In[26]:


# pip install imblearn


# In[27]:



from imblearn.over_sampling import SMOTE


# In[28]:


df['diagnosis'].value_counts()


# In[29]:


sm = SMOTE(random_state =42)
X_res, Y_res = sm.fit_resample (x_norm, y_norm)


# In[30]:


Y_res[0].value_counts()


# In[31]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res ,'Random Forest',RandomForestClassifier(), param, cv =10)


# In[32]:


param = {
            'C': [0.1,1,100,1000],
            'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res, Y_res,'SVC',SVC(), param, cv =5)


# In[33]:


param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res, Y_res,'XGBoost', XGBClassifier(),param, cv = 5)


# In[ ]:




