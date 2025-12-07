from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score,recall_score,confusion_matrix,f1_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statistics as stats
import seaborn as sns

dataset=pd.read_csv(r"C:\Users\Dell\OneDrive\python project\Food_Delivery_Time_Prediction.csv")
dataset=dataset.dropna()

label_cols=['Weather_Conditions','Traffic_Conditions','Vehicle_Type']

for col in label_cols:
    if col in dataset.columns:
        label_encoder=LabelEncoder()
        dataset[col]=label_encoder.fit_transform(dataset[[col]])
    else:
        print(f'column not found: {col}')

numeric_cols=['Distance','Delivery_Time','Order_Cost']

for col in numeric_cols:
    if col in dataset.columns:
        scaler=StandardScaler()
        dataset[col]=scaler.fit_transform(dataset[[col]])
    else:
        print(f'column not found: {col}')
        

numeric_feature=['Distance','Delivery_Person_Experience','Restaurant_Rating','Customer_Rating','Order_Cost','Tip_Amount']

for a in numeric_feature:
    if a in dataset.columns:
        mean=stats.mean(dataset[a])
        median=stats.median(dataset[a])
        mode=stats.mode(dataset[a])
        varience=stats.variance(dataset[a])
        print(f'Mean of {a} :', mean)
        print(f'Median of {a} :', median)
        print(f'Mode of {a} : ', mode)
        print(f'Varience of {a} :', varience)
    else:
        print(f'column not found : {a}')

for z in numeric_feature:
    if z in dataset.columns:
        plt.scatter(dataset[z],dataset['Delivery_Time'], color='blue', label='Data points')
        plt.xlabel(z)
        plt.ylabel('Delivery_Time')
        m,b=np.polyfit(dataset[z],dataset['Delivery_Time'],1)
        plt.plot(dataset[z],m*dataset[z]+b, color='red', label='Best-fit Line')
        plt.show()

plt.boxplot(dataset[z])
plt.title('Box Plot Example')
plt.ylabel('Value')
plt.show()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


median_time = dataset['Delivery_Time'].median()
dataset['Fast_Delivery'] = np.where(dataset['Delivery_Time'] < median_time, 1, 0)

#Phase2 (Linear Regression Model)

if 'Order_Time' in dataset.columns:
    dataset['Order_Time'] = dataset['Order_Time'].replace({
        'Morning': 0,
        'Afternoon': 1,
        'Evening': 2,
        'Night': 3
    })

if 'Order_Priority' in dataset.columns:
    dataset['Order_Priority'] = dataset['Order_Priority'].replace({
        'Low': 0,
        'Medium': 1,
        'High': 2
    })

if 'Traffic_Conditions' in dataset.columns:
    dataset['Traffic_Conditions'] = dataset['Traffic_Conditions'].replace({
        'Low': 0,
        'Medium': 1,
        'High': 2
    })

x=dataset[['Distance','Traffic_Conditions','Order_Priority']]
y=dataset['Delivery_Time']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lin_model=LinearRegression()

lin_model.fit(X_train,y_train)
lin_pred=lin_model.predict(X_test)
lin_pred=np.where(lin_model.predict(X_test)>=0.5,1,0)

print('---Result for linear Regression---')
print('\n')
print('R-square: ' , r2_score(y_test,lin_pred))
print('Mean Squared Error: ', mean_squared_error(y_test,lin_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test,lin_pred))

#(Logistic Regression Model)

if 'Weather_Conditions' in dataset.columns:
    dataset['Weather_Conditions'] = dataset['Weather_Conditions'].replace({
        'Sunny': 0,
        'Rainy': 1,
        'Cloudy': 2,
        'Snowy':3
    })

median_time=dataset['Delivery_Time'].median()
dataset['Fast_Delivery'] = np.where(dataset['Delivery_Time'] < median_time, 1, 0)

X=dataset[['Traffic_Conditions','Weather_Conditions','Delivery_Person_Experience']]
y=dataset['Fast_Delivery']

X_test_train,X_test_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train_train=sc.fit_transform(X_train)
X_test_test=sc.fit_transform(X_test)

log_model=LogisticRegression(random_state=0)

log_model.fit(X_train,y_train)

y_log_test=y_test

log_pred=log_model.predict(X_test)
print('\n')
print('---Evaluating required metrics--- ')

print("\n Accuracy:", accuracy_score(y_log_test,log_pred))
print("Precission:", precision_score(y_log_test,log_pred))
print("Recall:" , recall_score(y_log_test,log_pred))
print("F1score:" , f1_score(y_log_test,log_pred))
print("Confusion Matrix:" , confusion_matrix(y_log_test,log_pred))

#Phase3
# Comparing Linear Regression and Logistic Regression models

print('\n---For Linear Regression Model---')
print('\nAccuracy:', accuracy_score(y_test,lin_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test,lin_pred))

print('\n---For Logistic Regression Model---')
print("\n Accuracy:", accuracy_score(y_log_test,log_pred))
print(" Confusion Matrix:\n" , confusion_matrix(y_log_test,log_pred))

#Confusion Matrix

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, lin_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Linear Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

#ROC curv comparison

fpr_lin, tpr_lin, _ = roc_curve(y_test, lin_model.predict(X_test))
roc_auc_lin = auc(fpr_lin, tpr_lin)

fpr_log, tpr_log, _ = roc_curve(y_test, lin_model.predict(X_test))
roc_auc_log = auc(fpr_log, tpr_log)

plt.figure(figsize=(8,6))
plt.plot(fpr_lin, tpr_lin, label=f'Linear Regression (AUC = {roc_auc_lin:.2f})')
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})', linestyle='--')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

print("\n---Suggestions for optimal improvement---")
print('\n')
print('1)Using social media ads for better engagement.')
print('2)Optimizing delivery routes by provind GPS to delivery riders.')
print('3)Providing trainig to the cooking staff.')
print('4)Order price drops signifiently during snowy conditions so discounts could be offered.')

#Summary
print('\n')
print('---Summary---')
print('The model is made to predict delivery time based on different factors such as customer location, weather, traffic, and other factors. we made 2 different modals-')
print('a) Linear Regression Modal')
print('b)Logistic Regression Modal')
print('Firstly all the missing values from the data are dropped. proper encoding are done to different columns when the modals to pick the data since modals can only understand numeric values. Mean, median, mode and varience of all numeric columns are calculated and a scatter plot has also been formed between delivery time and different numeric colums.Box plot was also made and no as such outliers were found.')
print('Distace has also been calculates using Haversine Formula and time is also divided into rush and non rush hour.')
print('Different predictions using models are also made.')
print('Actionable insights and recommendations for optimizations are also made.')



