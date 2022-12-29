import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# Import data
df = pd.read_excel("Kickstarter.xlsx")

####### Pre-Processing
df = df.dropna()

#Data cleaning

df.drop(['id','name','pledged','deadline','state_changed_at', 'created_at', 'launched_at', 'static_usd_rate',
              'usd_pledged','name_len','blurb_len','created_at_weekday','deadline_day',
              'state_changed_at_weekday','state_changed_at_day',
              'state_changed_at_month','state_changed_at_yr', 'state_changed_at_hr',
              'created_at_month','created_at_day', 'created_at_yr', 'created_at_hr',
              'launched_at_day','create_to_launch_days','deadline_yr', 
              'launched_at_yr','staff_pick','backers_count',
              'disable_communication','currency', 'launch_to_state_change_days','spotlight'], axis=1, inplace=True )


#Remove 'canceled' and 'suspended' state
df.drop(df[df.state == 'canceled'].index, inplace=True)
df.drop(df[df.state == 'suspended'].index, inplace=True)

#Create independent and dependent variables
y= df['state']
y_one=pd.get_dummies(y)
y_two=pd.concat((y_one,y), axis=1)
y_two = y_two.drop(['state'],axis=1)
y_two=y_two.drop(['failed'], axis=1)
y = y_two.rename(columns={'successful': 'state'})
y=y['state']

X=df.iloc[:, np.r_[0:1,2:13]]

X_dummy = pd.get_dummies(X, columns=['country','category','deadline_weekday','launched_at_weekday'])

# Standardize predictors
scaler = StandardScaler()
X_std = scaler.fit_transform(X_dummy)
X_std=pd.DataFrame(X_std, columns=X_dummy.columns)

#Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.3, random_state = 0)

####Build model with Gradient Boosting Algorithm

# K-fold cross-validation with different number of samples required to split
for i in range (2,10):                                                                        
    model = GradientBoostingClassifier(random_state=0,min_samples_split=i,n_estimators=100)
    scores = cross_val_score(estimator=model, X=X_std, y=y, cv=5)
    print(i,':',np.average(scores))

#Build a GBT model with 6 samples required to split
gbt = GradientBoostingClassifier(random_state = 0, min_samples_split=6)
model_gbt = gbt.fit(X_train,y_train)
#pd.Series(model_gbt.feature_importances_,index=X.columns).sort_values(ascending = False).plot(kind = 'bar', figsize=(14,6))

# Make prediction and evaluate accuracy
y_test_pred = model_gbt.predict(X_test)
accuracy_gbt_full = accuracy_score(y_test, y_test_pred)
gbt_scores = np.average(cross_val_score(gbt, X=X_std, y=y,cv=5))

################################### Grading ################################

# Import Grading Data
grading_df = pd.read_excel("Kickstarter-Grading-Sample.xlsx")

# Pre-Process Grading Data
grading_df = grading_df.dropna()

grading_df.drop(['id','name','pledged','deadline','state_changed_at', 'created_at', 'launched_at', 'static_usd_rate',
              'usd_pledged','name_len','blurb_len','created_at_weekday','deadline_day',
              'state_changed_at_weekday','state_changed_at_day',
              'state_changed_at_month','state_changed_at_yr', 'state_changed_at_hr',
              'created_at_month','created_at_day', 'created_at_yr', 'created_at_hr',
              'launched_at_day','create_to_launch_days','deadline_yr', 
              'launched_at_yr','staff_pick','backers_count',
              'disable_communication','currency', 'launch_to_state_change_days','spotlight'], axis=1, inplace=True )
#The last 3 dropped because they are well distributed and does not show much difference btw failed and successful
#All values are the same for disable_communiation
#State change only after project is failed or successful --> no meaning for launch_to_state_change_days



grading_df.drop(grading_df[grading_df.state == 'canceled'].index, inplace=True)
grading_df.drop(grading_df[grading_df.state == 'suspended'].index, inplace=True)

#Create independent and dependent variables

# Setup the variables
X_grading = grading_df.iloc[:, np.r_[0:1,2:13]]
y_grading = grading_df["state"]

X_dummy = pd.get_dummies(X_grading, columns=['country','category','deadline_weekday','launched_at_weekday'])

scaler = StandardScaler()
X_grading_std = scaler.fit_transform(X_dummy)
X_grading_std=pd.DataFrame(X_grading_std, columns=X_dummy.columns)

# Apply the model previously trained to the grading data
gbt = GradientBoostingClassifier(random_state = 0, min_samples_split=9)
model_gbt = gbt.fit(X_grading_std,y_grading)

y_grading_pred = model_gbt.predict(X_grading_std)

# Calculate the accuracy score
accuracy_score(y_grading, y_grading_pred)
