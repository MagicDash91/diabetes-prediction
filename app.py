import streamlit as st
import pandas as pd
import numpy as np

st.header('Diabetes Prediction Using Random Forest Classifier')

df = pd.read_csv("diabetes.csv")
df_copy = df.copy(deep = True) #deep = True -> Buat salinan indeks dan data dalam dataframe
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df_copy['Glucose'].fillna(df_copy['Glucose'].median(), inplace = True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].median(), inplace = True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df_copy[(df_copy['Outcome']==0)] # semua data yang value outcome nya = 0
df_minority = df_copy[(df_copy['Outcome']==1)] # semua data yang value outcome nya = 1
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 n_samples= 500, # to match majority class, menyamakan jumlah value 1 dengan 0
                                 random_state=42)  # reproducible results, random state 0 is better than 42
                                                  #Random state = Mengontrol pengacakan yang diterapkan ke data agar hasil yang didapatkan tetap sama
# Combine majority class with upsampled minority class
df_copy2 = pd.concat([df_minority_upsampled, df_majority])

import scipy.stats as stats
z = np.abs(stats.zscore(df_copy2))
data_clean = df_copy2[(z<3).all(axis = 1)] #print all of rows that have z<3 (z score below 3)

X = data_clean.drop('Outcome', axis=1) #menggunakan semua atribut kecuali class (Outcome)
y = data_clean['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=0)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0) #Mengontrol keacakan estimator agar hasil yang didapatkan selalu tetap
clf.fit(X_train, y_train)

preg = st.number_input('Pregnancies')
gluc = st.number_input('Glucoses')
blood = st.number_input('Blood Pressure')
skin = st.number_input('Skin Thickness')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
diabet = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

Xnew3 = [[preg,gluc,blood,skin,insulin,bmi,diabet,age]]

y_pred_prob4 = clf.predict_proba(Xnew3)
y_pred_prob_df4 = pd.DataFrame(data=y_pred_prob4, columns=['Prob of dont have diabetes', 'Prob of have diabetes'])
hasil = (y_pred_prob_df4)*100
st.write("Result")
st.dataframe(hasil)
st.write("Bar Chart Result")
st.bar_chart(hasil)
