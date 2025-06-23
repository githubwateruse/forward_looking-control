# -*- coding: utf-8 -*-
"""
Created on Wed May 14 03:41:47 2025

@author: sp825
"""


""" 
# Neural network for mapping the historical rainfall data to the future tank levels.

# The model learns the relationship between past rainfall data (as well as current upstream node depths) and the future water levels of the tanks.

The data used for training and testing is generated in the file: 
#   '1. Training and testing datasets_ rainfall data vs tank level.py'
"""


import numpy as np
import xlrd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib  

from datetime import datetime

start_time = datetime.now()



def dataset_extraction(filename):
    
    DATA = []

    data_path_name = "C:/xiaoxin backup/end-to-end control/Astlingen simulation/rainfall_tanklevel_datasets/rainfall_tanklevel_" + filename + ".xlsx"
    
    workbook = xlrd.open_workbook(data_path_name)
    data_sheet = workbook.sheet_by_index(0) #通过索引获取该列数据
    rowNum = data_sheet.nrows
    colNum = data_sheet.ncols

    
    
    for mmm in range (rowNum):
        sample = data_sheet.row_values(mmm)
        DATA.append(sample)
    
    
        
    return np.array(DATA)




X_train = dataset_extraction("X_train")
X_test = dataset_extraction("X_test")


Y_train = dataset_extraction("Y_train")
Y_test = dataset_extraction("Y_test")



param_grid = {
    'hidden_layer_sizes': [(100, 100)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate_init': [0.001, 0.003, 0.005,
                           0.007, 0.009, 0.01, 0.011, 0.013],
}


mlp = MLPRegressor(max_iter=1000, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# best model
best_model = grid_search.best_estimator_


Y_pred = best_model.predict(X_test)

model = best_model
print("Train score: ", model.score(X_train, Y_train))
print("Test score: ", model.score(X_test, Y_test))

# saving models
model_filename = 'rainfall_tanklevel_datasets/rainfall_data_vs_tanklevel_30min_advance.pkl'
joblib.dump(model, model_filename)

# loading models
loaded_model = joblib.load(model_filename)


loaded_Y_pred = loaded_model.predict(X_test)
end_time = datetime.now()
print("calculation time", end_time - start_time)