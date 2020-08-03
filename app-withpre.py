# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:14:19 2020

@author: srinath.mugundhan
"""
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import pickle
import joblib
from datetime import date
import calendar
import datetime
import traceback

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)
            query['Unique']=query['CPU']+query['Invoice #']
            query['Last Action Code']=query['Last Action Code'].astype(str)
            query['Last Denial Reason']=query['Last Denial Reason'].astype(str)
            unq=query.groupby('Unique')['Last Action Code'].min().reset_index()
            unq1=query.groupby('Unique')['Last Denial Reason'].min().reset_index()
            query1 = query[['CPU','Invoice #','Payor Role','Reporting Payor Name','Inv Payor Open Amt','Service Date','Last Action Create Date']] 
            query1['Payor Role'] = query1[['Payor Role']].replace(to_replace =' ',value =np.nan) 
            query1['Payor Role'] = query1[['Payor Role']].astype(float)
            query1['Unique'] = query1['CPU']+query1['Invoice #']
            query1 = query1 .drop(['CPU','Invoice #'],axis=1)           
            def transform_cdt2(df):
                create_date = list()
                for i in df['Service Date']:
                    if (type(i)== int):
                        i = datetime.date.fromordinal(693594 + i)
                        create_date.append(i)
                    else:
                        create_date.append(i)
                df['Service Date'] = create_date
                df['Service Date'] = df['Service Date'].astype(np.datetime64)
                return df
            query1 = transform_cdt2(query1)
            def transform_cdt2(df):
                create_date = list()
                for i in df['Last Action Create Date']:
                    if (type(i)== int):
                        i = datetime.date.fromordinal(693594 + i)
                        create_date.append(i)
                    else:
                        create_date.append(i)
                df['Last Action Create Date'] = create_date
                df['Last Action Create Date'] = df['Last Action Create Date'].astype(np.datetime64)
                return df
            query1 = transform_cdt2(query1)
            query1['current_date'] = pd.to_datetime('today').date()
            ag = query1[['Unique','Service Date','current_date','Last Action Create Date']]
            novinvdt = query1[['Unique','Service Date','current_date','Last Action Create Date']]
            x=query1.groupby('Unique')['Inv Payor Open Amt'].count().reset_index()
            x.rename(columns = { 'Inv Payor Open Amt':'pretouch_time'}, inplace = True)
            y=query1.groupby('Unique')['Inv Payor Open Amt'].max().reset_index()
            y.rename(columns = { 'Inv Payor Open Amt':'PreTouch_Inv_Balance'}, inplace = True)
            z=query1.groupby('Unique')['Payor Role'].min().reset_index()
            z['Payor Role'] = z['Payor Role'].astype(str)
            r = query1.groupby('Unique')['Reporting Payor Name'].min().reset_index()
            val = (pd.merge(x, y, on='Unique',how='left'))
            val = (pd.merge(val, z, on='Unique',how='left'))
            val = (pd.merge(val, r, on='Unique',how='left'))
            ag.rename(columns = { 'Last Action Create Date':'Last Action Day'}, inplace = True)
            ag = ag.groupby('Unique')['Last Action Day','current_date'].min().reset_index()
            val = (pd.merge(val, ag, on='Unique',how='left'))
            novinvdt.rename(columns = { 'Service Date':'Inv_Date'}, inplace = True)
            novinvdt1 = novinvdt[['Unique','Inv_Date']]
            novinvdt1 = novinvdt1.groupby('Unique')['Inv_Date'].min().reset_index()
            val = (pd.merge(val, novinvdt1, on='Unique',how='left'))
            map.rename(columns = { 'Payor Name':'Reporting Payor Name'}, inplace = True)
            val=(pd.merge(val, map, on='Reporting Payor Name',how ='left'))
            val['Payor Grouping'].fillna("other", inplace = True)
            val['Dt_LacDt'] = val['current_date'].astype(np.datetime64)-val['Last Action Day'].astype(np.datetime64)
            val['Dt_LacDt'] = val['Dt_LacDt'].dt.days.astype('str')
            val['Dt_LacDt'] = val['Dt_LacDt'].map(lambda x: x.lstrip('+-').rstrip('days'))
            val['Invc_age'] = val['Inv_Date'].astype(np.datetime64)-val['current_date'].astype(np.datetime64)
            val['Invc_age'] = val['Invc_age'].dt.days.astype('str')
            val['Invc_age'] = val['Invc_age'].map(lambda x: x.lstrip('+-').rstrip('days'))
            val['Invc_age'] = val['Invc_age'].astype(float)
            val['Dt_LacDt']=val['Dt_LacDt'].astype(float)
            val['Payor Role']=val['Payor Role'].astype(float)
            val29 = val.copy()
            onehot = pd.get_dummies(val['Payor Grouping'])
            val = pd.concat([val,onehot],axis=1)
            val = val.drop(['Reporting Payor Name','Unique','Payor Grouping','Last Action Day','Inv_Date','current_date'],axis=1)
            val['Payor Role'].fillna(value =val['Payor Role'].median(),inplace = True)
            val['PreTouch_Inv_Balance'].fillna(value =val['PreTouch_Inv_Balance'].median(),inplace = True)
            val['Invc_age'].fillna(value =val['Invc_age'].median(),inplace = True)
            val['Dt_LacDt'].fillna(value =val['Dt_LacDt'].median(),inplace = True)          
            l_column = ['Payor Role','PreTouch_Inv_Balance','pretouch_time',
                      'Dt_LacDt','Invc_age','AARP','Aetna','Ambetter',
                      'AmeriHealth','Anthem','BCBS','BS CA','CCX',
                      'Cigna','Commercials','Commercials ','Contra Costa',
                      'Fidelis','First Health','Group Health','HIP',
                      'HMSA','Health Net','Humana','John Muir','Kaiser',
                      'Keystone','Medicaid Plans','Medicaid Plans ',
                      'Medicare Plans','Omaha','Premera','Private Healthcare',
                      'Regence','Supermed','UHC','Wellcare','other']
            for i in l_column:
                if i not in val.columns:
                    val[i]=0
            new_val=val[val.columns.intersection(l_column)]
            #new_val = pd.get_dummies(new_val)
            new_val = new_val.reindex(columns=model_columns, fill_value=0)
            prediction = lr.predict_proba(new_val)
            prediction = pd.DataFrame(prediction)
            prediction.rename(columns = {0 :'propensity_0',1:'propensity_1'}, inplace = True)
            prediction.loc[(prediction.propensity_1 > 0.5),'HMLR']='High'
            prediction.loc[(prediction.propensity_1 > 0.3) & (prediction.propensity_1 < 0.5),'HMLR']='Medium'
            prediction.loc[(prediction.propensity_1 > 0.15) & (prediction.propensity_1 < 0.3),'HMLR']='Low'
            prediction.loc[(prediction.propensity_1 > 0) & (prediction.propensity_1 < 0.15),'HMLR']='Res'
            prediction1=pd.concat([val29,prediction],axis=1)
            prediction1=prediction1.drop(['propensity_0','Last Action Day','current_date','Inv_Date',
                                          'Dt_LacDt','Invc_age'],axis=1)
            prediction1=(pd.merge(prediction1,unq, on='Unique',how='left'))
            prediction1=(pd.merge(prediction1,unq1, on='Unique',how='left'))
            prediction1=prediction1[['Unique','Payor Role','Reporting Payor Name','PreTouch_Inv_Balance',
                                         'Last Action Code','Last Denial Reason','propensity_1','HMLR']]
            prediction1=prediction1.to_json(orient ='records')
            #prediction = list(prediction['propensity_1'])
            #Tag=list(prediction['HMLR'])            
            #return jsonify({'Inout':str(query)},{'prediction': str(prediction1)},{'Tag':str(Tag)})
            return jsonify({'Response':str(prediction1)})
        
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 5000 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model_LR.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    map=pd.read_excel('paygrop_map.xlsx')
    print('payorgrouping file loaded')
    
    app.run(port=port, debug=False)