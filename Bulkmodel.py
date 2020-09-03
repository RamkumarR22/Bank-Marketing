# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:39:20 2020

@author: Ram RS
"""
import streamlit as st
#import itertools
from PIL import Image
# EDA Pkgs
import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from lightgbm import LGBMClassifier
import base64
#from io import BytesIO
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pybase64


#%matplotlib inline
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#import warnings
#warnings.filterwarnings('ignore')

LGBM_model = pickle.load(open("finalized_LGBM.sav", 'rb'))

st.set_option('deprecation.showfileUploaderEncoding', False)


def Numerical_variables(x):
    Num_var = [var for var in x.columns if x[var].dtypes!="object"]
    Num_var = x[Num_var]
    return Num_var

def categorical_variables(x):
    cat_var = [var for var in x.columns if x[var].dtypes=="object"]
    cat_var = x[cat_var]
    return cat_var

def impute(x):
    df=x.dropna()
    return df

def Show_pearsonr(x,y):
    result = pearsonr(x,y)
    return result

from scipy.stats import spearmanr
def Show_spearmanr(x,y):
    result = spearmanr(x,y)
    return result

import plotly.express as px
def plotly(x,y):
    sns.scatterplot(x,y)
    #plt.show()

def show_displot(x):
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(x)


        plt.subplot(122)
        x.plot.box(figsize=(16,5))

        plt.show()

def Show_DisPlot(x):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,7))
    return sns.distplot(x, bins = 25)

def Show_CountPlot(x):
    fig_dims = (18, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    return sns.countplot(x,ax=ax)

def plotly_histogram(x,y):
    sns.barplot(x,y)
    #plt.show()

def plotly_violin(x,y):
    sns.violinplot(x,y)



from scipy import stats
def Tabulation(x):
    table = pd.DataFrame(x.dtypes,columns=['dtypes'])
    table1 =pd.DataFrame(x.columns,columns=['Names'])
    table = table.reset_index()
    table= table.rename(columns={'index':'Name'})
    table['No of Missing'] = x.isnull().sum().values    
    table['No of Uniques'] = x.nunique().values
    table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
    table['First Observation'] = x.loc[0].values
    table['Second Observation'] = x.loc[1].values
    table['Third Observation'] = x.loc[2].values
    for name in table['Name'].value_counts().index:
        table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
    return table


def Show_PairPlot(x):
    return sns.pairplot(x)

def Show_HeatMap(x):
    f,ax = plt.subplots(figsize=(15, 15))
    return sns.heatmap(x.corr(),annot=True,ax=ax);
def show_dtypes(x):
    return x.dtypes

def show_columns(x):
    return x.columns

def Show_Missing(x):
    return x.isna().sum()
def Show_Missing1(x):
    return x.isna().sum()

def Show_Missing2(x):
    return x.isna().sum()

def show_hist(x):
    return x.hist()

def ShowVisuals(df):
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(13,9))

 

    df["Age_Group"] = df["age"]
    df.loc[(df["age"] >= 18) & (df["age"] < 38), "Age_Group"] = 0
    df.loc[(df["age"] >= 38), "Age_Group"] = 1

 

    #replacing those numbers to categorical features then get the dummy variables
    df["Age_Group"] = df["Age_Group"].replace(0, "Age<38")
    df["Age_Group"] = df["Age_Group"].replace(1, "Age>38")

 


    sns.barplot(y=df["age"], x=df["y"], ax= axs[0][0])
    sns.barplot(x=df["y"], y=df["campaign"], ax= axs[0][1])
    sns.barplot(x=df["y"], y=df["emp.var.rate"], ax= axs[1][0])
    sns.barplot(x=df["y"], y=df["euribor3m"], ax= axs[1][1])


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = pybase64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="Prediction.csv">⬇️ Download output CSV File</a>'



def preprocess(data):
    KeepColumns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m']
    df = data[KeepColumns]
    
    #replace unknown for job,education
    df.loc[(df['age']>60) & (df['job']=='unknown'), 'job'] = 'retired'
    df.loc[(df['education']=='unknown') & (df['job']=='management'), 'education'] = 'university.degree'
    df.loc[(df['education']=='unknown') & (df['job']=='services'), 'education'] = 'high.school'
    df.loc[(df['education']=='unknown') & (df['job']=='housemaid'), 'education'] = 'basic.4y'
    df.loc[(df['job'] == 'unknown') & (df['education']=='basic.4y'), 'job'] = 'blue-collar'
    df.loc[(df['job'] == 'unknown') & (df['education']=='basic.6y'), 'job'] = 'blue-collar'
    df.loc[(df['job'] == 'unknown') & (df['education']=='basic.9y'), 'job'] = 'blue-collar'
    df.loc[(df['job']=='unknown') & (df['education']=='professional.course'), 'job'] = 'technician'
    
    df = df.replace("unknown",np.nan)
    
    
    #Grouping all 4 to 9years basic to basic_general
    df["education"] = df["education"].replace(["basic.4y","basic.6y","basic.9y"],"basic")
    
    #Ordinally encoding the education variable
    df["education"] = df["education"].replace(["illiterate","basic","high.school","university.degree",
                                                 "professional.course"],[0,1,2,3,4])
    
    #Renameing variable name
    df.rename(columns={"default":"Credit_Default","housing":"housing_loan","loan":"personal_loan"},inplace=True)

    
    # Encoding Credit default, housing loan and personbal loan variables

    CreditDefaultMap = {"no":1,
                        "yes":0}
    
    HouseLoanMap = {"no":1,
                    "yes":0}
    
    PersonalMap = {"no":1,
                   "yes":0}
    
    
    df["Credit_Default"] = df["Credit_Default"].map(CreditDefaultMap)
    df["housing_loan"] = df["housing_loan"].map(HouseLoanMap)
    df["personal_loan"] = df["personal_loan"].map(PersonalMap)
    
    #Grouping status
    df["job"] = df["job"].replace(["admin.","blue-collar",'technician','services', 'management'],"Employeed")
    df["job"] = df["job"].replace(['entrepreneur', 'self-employed'],"Business")
    df["job"] = df["job"].replace(['unemployed','student',"housemaid"],"Unemployeed")
    
    df["job"] = df["job"].replace(["Business","Employeed","Unemployeed",
                                             "retired"],[3,2,0,1])
    
    
    
    df.rename(columns = {"emp.var.rate": "EmployeeVariationRate",
                     "cons.price.idx":"ConsumerPriceIndex",
                     "cons.conf.idx": "ConsumerConfidenceIndex"}, inplace = True)
    
    df = pd.get_dummies(df, drop_first=True)
    
    #imputation
    imputer = KNNImputer(n_neighbors=3) #Using N-Neighbours as 3
    imputer_array = imputer.fit_transform(df)
    df = pd.DataFrame(imputer_array, columns  = df.columns)
    
    #x,y
    X = df.copy()
    #Y = Y.replace(["no","yes"],[0,1])
    
    pred=LGBM_model.predict(X)
    pred_prob = LGBM_model.predict_proba(X)
    #Likelihood = []
    Likelihood1 = []
    for i,j in zip(pred_prob[:,0],pred_prob[:,1]):
        if i > 0.55:
            #Likelihood.append(1)
            Likelihood1.append(i)
        else: 
            #Likelihood.append(0)
            Likelihood1.append(j)
            
    
    #p = prediction[:,1]
    #if p > 0.5:
     #   st.warning('{}% Chance of risk'.format((p)*[100]))
    #else:
     #   st.success('{}% Chance of risk'.format((p)*[100]))
    
    Predictions = pd.DataFrame(pred,columns=["Term_deposit"])
    Predictions["Term_deposit"] = Predictions["Term_deposit"].replace([0,1],["No","Yes"])
    Predictions["Likelihood_of_customer"] = np.round(Likelihood1,3)
    Predictions = pd.concat([data,Predictions],axis=1)
    st.dataframe(Predictions.head(10))
    #st.markdown('### **⬇️ Download output CSV File **')
    st.markdown(get_table_download_link(Predictions), unsafe_allow_html=True)
    #output = BytesIO()
    #writer = pd.ExcelWriter(output, engine='xlsxwriter')
    #Predictions.to_excel(writer, sheet_name='Sheet1')
    #writer.save()
    #processed_data = output.getvalue()
    #val = processed_data
    #b64 = pybase64.b64encode(val)
    #st.dataframe(Predictions.head(10))
    #st.success(Predictions["Term_deposit"].value_counts())
    #st.success("Prediction is completed, Download the result using below link")
    #st.markdown(f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Result.xlsx">Download Result file</a>', unsafe_allow_html=True)
    #csv = Predictions.to_csv(index=False)
    #b64 = base64.b64encode(csv.encode()).decode()
    #st.markdown('### **⬇️ Download output CSV File **')
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ".csv")' 
    #st.markdown(href, unsafe_allow_html=True)
    
    
    
def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">Bank Marketing ML App</h1>
    </div>
    """
    #st.header("Know whether the customer will accept 'Term deposit' or Not")
    #st.markdown(html_temp,unsafe_allow_html=True)
    activities = ["General EDA","Insightful EDA","Prediction"]	
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    
    if choice == "General EDA":
        st.image("nttlogo-blue.png",width=350,use_cloumn_width=2000,clamp=True)
        
        data = st.file_uploader("Feed the Data here..", type = ["xlsx","csv","txt"] ,
                            showfileUploaderEncoding=False)
        
        if data is not None:
            df = pd.read_csv(data,sep=';')
            KeepColumns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m']
            df = df[KeepColumns]
            st.success("Data Frame Loaded successfully")
            
            if st.checkbox("Numerical Variables"):
                num_df = Numerical_variables(df)
                numer_df=pd.DataFrame(num_df)
                st.dataframe(numer_df.head(10))
            if st.checkbox("Categorical Variables"):
                new_df = categorical_variables(df)
                catego_df=pd.DataFrame(new_df)
                st.dataframe(catego_df.head(10))
            cat = [col for col in df.columns if df[col].dtype=='object']
            cat_df = df[cat]
            num_df = df.drop(cat,axis=1)
            
            
            
            st.subheader('Pearsonr Correlation')
            all_columns_names3 = show_columns(num_df)
            all_columns_names4 = show_columns(num_df)
            selected_columns_name3 = st.selectbox("Select Column 1 For Pearsonr Correlation (Numerical Columns)",all_columns_names3)
            selected_columns_names4 = st.selectbox("Select Column 2 For Pearsonr Correlation (Numerical Columns)",all_columns_names4)
            if st.button("Generate Pearsonr Correlation"):
                pr=pd.DataFrame(Show_pearsonr(num_df[selected_columns_name3],num_df[selected_columns_names4]),index=['Pvalue', '0'])
                st.dataframe(pr)
            
            
            st.subheader("UNIVARIATE ANALYSIS")
            
            all_columns_names = show_columns(num_df)
            selected_columns_names = st.selectbox("Select Column for Distplot ",all_columns_names)
            if st.checkbox("Show DisPlot for Selected variable"):
                st.write(Show_DisPlot(num_df[selected_columns_names]))
                st.pyplot()
            all_columns_names = show_columns(cat_df)
            selected_columns_names = st.selectbox("Select Column for CountPlot ",all_columns_names)
            if st.checkbox("Show CountPlot for Selected variable"):
                st.write(Show_CountPlot(cat_df[selected_columns_names]))
                st.pyplot()
            
            
            
            st.subheader("BIVARIATE ANALYSIS")
            Scatter1 = show_columns(num_df)
            Scatter2 = show_columns(num_df)
            Scatter11 = st.selectbox("Select Column 1 For Scatter Plot (Numerical Columns)",Scatter1)
            Scatter22 = st.selectbox("Select Column 2 For Scatter Plot (Numerical Columns)",Scatter2)
            if st.button("Generate PLOTLY Scatter PLOT"):
                st.pyplot(plotly(num_df[Scatter11],num_df[Scatter22]))
            
            
            bar1 = show_columns(num_df)
            bar2 = show_columns(cat_df)
            bar11 = st.selectbox("Select Column 1 For Bar Plot ",bar1)
            bar22 = st.selectbox("Select Column 2 For Bar Plot ",bar2)
            if st.button("Generate PLOTLY BAR PLOT"):
                st.pyplot(plotly_histogram(df[bar11],df[bar22]))
            
            
            violin1 = show_columns(cat_df)
            violin2 = show_columns(num_df)
            violin11 = st.selectbox("Select Column 1 For violin Plot",violin1)
            violin22 = st.selectbox("Select Column 2 For violin Plot",violin2)
            if st.button("Generate PLOTLY violin PLOT"):
                st.pyplot(plotly_violin(df[violin11],df[violin22]))
            
            
            st.subheader("MULTIVARIATE ANALYSIS")
            if st.checkbox("Show Histogram"):
                st.write(show_hist(num_df))
                st.pyplot()
            if st.checkbox("Show HeatMap"):
                st.write(Show_HeatMap(df))
                st.pyplot()
            if st.checkbox("Show PairPlot"):
                st.write(Show_PairPlot(df))
                st.pyplot()
    
    if choice == "Insightful EDA":
        st.image("nttlogo-blue.png",width=350,use_cloumn_width=2000,clamp=True)
        data = st.file_uploader("Start Your Experiments", type = ["xlsx","csv","txt"] ,
                            showfileUploaderEncoding=False)
        if data is not None:
            st.success("Dataset Uploaded Successfuly")
            
            df = pd.read_csv(data,sep=';')
            if st.button("Visualize"):
                st.pyplot(ShowVisuals(df))
            
    
    
    if choice== "Prediction":
        st.image("nttlogo-blue.png",width=350,use_cloumn_width=2000,clamp=True)
        data = st.file_uploader("Start Your Experiments", type = ["xlsx","csv","txt"] ,
                            showfileUploaderEncoding=False)
        
        if data is not None:
            st.success("Dataset Uploaded Successfuly")
            
            df = pd.read_csv(data,sep=';')
            st.dataframe(df.head())
            
        if st.button('Predict'):
            st.write("Please wait while loading....")
            preprocess(data=df)
            #st.markdown(get_table_download_link(Predictions), unsafe_allow_html=True)
            
        
        
        
        
        

if __name__ == '__main__':
    main()