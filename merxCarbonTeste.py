# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:52:34 2022

@author: gabriel.ferraz
"""
# =============================================================================
# 1. Seleção das variáveis explicativas (features);
#     Como pode ser visto no código, foram escolhidas as bandas azul, vermelho e IVP,
#     que compõe o índice EVI (Enhanced Vegetation Index). Isso se deve ao fato
#     de as bandas B4 e B8 combinadas tem bons resultados ao discernir vegetação
#     de área não vegetada. Isso poderia ser atingido apenas com o índice NDVI,
#     porém a adição da banda do azul no cálculo tem o poder de diferenciar alvos
#     pelo índice de área foliar (IAF) e arquitetura do dossel, por exemplo.
# =============================================================================

# =============================================================================
# 2. Seleção e construção do modelo de Machine Learning;
# Foi eleito o método "random forest" por este ser bastante utilizado em modelos
# de classificação. O modelo foi capaz de parametrizar o treinamento de acordo com
# resultados que ele obteve testando com os seguintes atribuitos:
#     
    #param_grid = {
    #                 "max_depth": [8, 9, 10, 15, 20, 25, 30],
    #                 "max_features": ["sqrt", "log2"],
    #                 "n_estimators": [100, 200, 300, 500],
    #                 "criterion": ["gini", "entropy"], 
    #                 "class_weight": ["balanced", "balanced_subsample"]
    #             }
#     
# O método RandomizedSearchCV do pacote sklearn otimiza o teste por validação cruzada
# e scores de qualidade internos ao método para obter o melhor modelo possível.
# 
# O dataset foi dividido em treino (75%), teste (10%) e validação (15%) 
# =============================================================================

# =============================================================================
# 3. Avaliação da performance do modelo;
# Será possível avaliar a performance do modelo através das seguintes métricas:
#     - score de acurácia
#     - Mean Absolute Error (MAE)
#     - Mean Squared Error (MSE)
#     - Root Mean Squared Error (RMSE)
#     - Mean Absolute Percentage Error (MAPE)
#     - Explained Variance Score
#     - Max Error
#     - Mean Squared Log Error
#     - Median Absolute Error
#     - R^2
# 
# Todos em cima do grupo de validação, escolhido por ser maior que o de teste. 
# =============================================================================

# =============================================================================
# 4. Avalição da importância das features.
# Existem estudos que compravam a capacidade das bandas 11 e 12
# (diferentes expectros de Short Wave Infra-Red) de discernir culturas agrícolas
# presentes em campo na mesma época. No entanto, isso aumentaria a complexidade
# do modelo para um ganho possivelmente marginal na tentativa de classifcar apenas
# "TRUE" ou "FALSE" para determinada cultura, não classificar alvos em diferentes
# culturas. Dessa forma entende-se que o índice EVI já coporta diversidade de
# informação capaz de atender ao objetivo.
# =============================================================================
                                                              


import ee
from datetime import datetime as dt
import pandas as pd
import numpy as np
import pygeohash as pgh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import save_log
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

ee.Authenticate()
ee.Initialize()

class MerxCarbonIdentifyCulture:
    def __init__(self):
        self.start = ee.Date('2021-08-01')
        self.end = ee.Date('2021-12-31')
        self.CloudCover = 20
        self.samples = ee.FeatureCollection('users/user/samples')
       
    
    def getEVI(self, image):
        # Compute the EVI using an expression.
        EVI = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000),
                'BLUE': image.select('B2').divide(10000)
            }).rename("EVI")
    
        image = image.addBands(EVI)
    
        return(image)
    
    def getPointValue(self, point):
        
        aoi = ee.Point(point.get("X"), point.get("Y"))
        
        imagecol = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(aoi).filterDate(self.start, self.finish).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',self.CloudCover)).filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT',self.CloudCover))
        
        images = [item.get('id') for item in imagecol.getInfo().get('features')]
            
        store = []
        date_store = []
        
        for image in images:
                    
            im = ee.Image(image)
            
            eviImage = self.getEVI(image)
            
            date = dt.fromtimestamp(im.get("system:time_start").getInfo() / 1000.)
            date_store.append(np.datetime64(date))
            
            projection = im.projection().getInfo()['crs']**im.projection().getInfo()['crs']
            data = eviImage.select("EVI").reduceRegion(ee.Reducer.first(),point,1,crs=projection).get("EVI")
            
            store.append(data.getInfo())
            
        
        
        df = pd.DataFrame(index = pgh.encode(point.get("X"), point.get("Y"), precision=5),data=store, columns=[date_store])
        df["flag"] = point.get("CULTURA")
        return df
    
    def convert_class_to_boolean(self, df, field, str_0, str_1):
        boolean_condition = df['{}'.format(field)] == '{}'.format(str_0)
        column_name = '{}'.format(field)
        new_value = '0'
        df.loc[boolean_condition, column_name] = new_value
    
        boolean_condition = df['{}'.format(field)] == '{}'.format(str_1)
        column_name = '{}'.format(field)
        new_value = '1'
        df.loc[boolean_condition, column_name] = new_value
        
    #Análise com RandomForestClassifier Otimizado - Treino, Teste e Validação
    def random_forest_classifier_custom_train_test_validation(data_analysis, fileFullName):
        try:
            param_grid = {
                "max_depth": [8, 9, 10, 15, 20, 25, 30],
                "max_features": ["sqrt", "log2"],
                "n_estimators": [100, 200, 300, 500],
                "criterion": ["gini", "entropy"], 
                "class_weight": ["balanced", "balanced_subsample"]
            }
    
            train_ratio = 0.75
            validation_ratio = 0.15
            test_ratio = 0.10
            X, y = data_analysis[:, 2:int(len(data_analysis))], data_analysis[:, 1]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    
            #Definir parâmetros de otimização   
            cf = RandomForestClassifier()
            cf_cv = RandomizedSearchCV(estimator=cf, param_distributions=param_grid, n_iter = 100, cv=3, verbose=5, random_state=42, n_jobs = -1)
            # cf_cv = GridSearchCV(estimator=cf, param_grid=param_grid, cv=3, verbose=5)
            cf_cv.fit(x_train, y_train)
            cf_cv.best_params_
                     
            #Executar RandomForestClassifier
            modelo=RandomForestClassifier(random_state=42, class_weight=cf_cv.best_params_['class_weight'], criterion=cf_cv.best_params_['criterion'], max_features=cf_cv.best_params_['max_features'], n_estimators=cf_cv.best_params_['n_estimators'], max_depth=cf_cv.best_params_['max_depth'])
            modelo.fit(x_train, y_train)
    
            y_pred=modelo.predict(x_test)
            new_y_pred = []
            for i in y_pred.tolist(): 
                if i == '0':
                    new_y_pred.append(0) 
                if i == '1':
                    new_y_pred.append(1)
    
            new_y_test = []
            for i in y_test.tolist(): 
                if i == '0':
                    new_y_test.append(0) 
                if i == '1':
                    new_y_test.append(1)  
          
    
            y_pred_val=modelo.predict(x_val)
            new_y_pred_val = []
            for i in y_pred_val.tolist(): 
                if i == '0':
                    new_y_pred_val.append(0) 
                if i == '1':
                    new_y_pred_val.append(1)
    
            new_y_val = []
            for i in y_val.tolist(): 
                if i == '0':
                    new_y_val.append(0) 
                if i == '1':
                    new_y_val.append(1)
                    
            save_log.saveTextLog(fileFullName, "Modelo otimizado accuracy_score ==> VALIDAÇÃO " + str(metrics.accuracy_score(new_y_val,new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Mean Absolute Error (MAE) ==> VALIDAÇÃO ' + str(metrics.mean_absolute_error(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Mean Squared Error (MSE) ==> VALIDAÇÃO ' + str(metrics.mean_squared_error(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Root Mean Squared Error (RMSE) ==> VALIDAÇÃO ' + str(metrics.mean_squared_error(new_y_val, new_y_pred_val, squared=False)))
            save_log.saveTextLog(fileFullName, 'Mean Absolute Percentage Error (MAPE) ==> VALIDAÇÃO ' + str(metrics.mean_absolute_percentage_error(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Explained Variance Score ==> VALIDAÇÃO ' + str(metrics.explained_variance_score(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Max Error ==> VALIDAÇÃO ' + str(metrics.max_error(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Mean Squared Log Error ==> VALIDAÇÃO ' + str(metrics.mean_squared_log_error(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'Median Absolute Error ==> VALIDAÇÃO ' + str(metrics.median_absolute_error(new_y_val, new_y_pred_val)))
            save_log.saveTextLog(fileFullName, 'R^2 ==> VALIDAÇÃO ' + str(metrics.r2_score(new_y_val, new_y_pred_val)))
           
            return modelo    
        except Exception as e:
            return e
        
    def save_model(self, model_name, model_file_name):
        filename = os.path.join(os.path.dirname(__file__), '{}.sav'.format(model_file_name))
        joblib.dump(model_name, filename)
        
    def execute(self):
        finalDf = pd.DataFrame()
        
        for point in self.samples:
            parcialDf = self.getPointValue(point, 20)
            finalDf = finalDf.append(parcialDf)
        
        self.convert_class_to_boolean(finalDf, 'CULTURA', 'nao_cultura', 'cultura')
        data_analysis = finalDf.values
        modelo = self.random_forest_classifier_custom_train_test_validation(data_analysis, "log.txt")
        self.save_model(modelo, "modelo_identificacao_cultura")

if __name__=="__main__":
    m = MerxCarbonIdentifyCulture()
    m.execute()