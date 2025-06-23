# -*- coding: utf-8 -*-
"""
Created on Wed May 14 01:46:38 2025

@author: sp825
"""


""" 
This .py file is used for training and testing dataset split

Timestep is 5 minutes

X_sample: Data from RG1, RG2, RG3, and RG4 over the past 30 minutes, 
          as well as the current depth of J13, J12, J4, and J8

Y_sample: Depth of T2, T3, T4, and T6 at 30 minutes in the future
""" 

import xlsxwriter
import xlrd
import numpy as np
from sklearn.model_selection import train_test_split




def dataset_(filename_, data_):
        
    workbook_params = xlsxwriter.Workbook('rainfall_tanklevel_datasets/' + 'rainfall_tanklevel_'+filename_ +'.xlsx')
    worksheet_params = workbook_params.add_worksheet('sheet1')

    for nnn in range (len(data_)):
        worksheet_params.write_row(nnn, 0, data_[nnn])

    workbook_params.close()
    
    
    
    
data_path_name = "C:/xiaoxin backup/end-to-end control/Astlingen simulation/rainfall_tanklevel_datasets/data_collection.xlsx"

workbook = xlrd.open_workbook(data_path_name)
data_sheet = workbook.sheet_by_index(0) #通过索引获取该列数据
rowNum = data_sheet.nrows
colNum = data_sheet.ncols
# print(rowNum)

RG1_data = np.array(data_sheet.col_values(0))
RG2_data = np.array(data_sheet.col_values(1))
RG3_data = np.array(data_sheet.col_values(2))
RG4_data = np.array(data_sheet.col_values(3))


J13_data = np.array(data_sheet.col_values(5))
J12_data = np.array(data_sheet.col_values(6))
J4_data = np.array(data_sheet.col_values(7))
J8_data = np.array(data_sheet.col_values(8))


T2_data = np.array(data_sheet.col_values(10))
T3_data = np.array(data_sheet.col_values(11))
T4_data = np.array(data_sheet.col_values(12))
T6_data = np.array(data_sheet.col_values(13))


# Rainfall inputs lead water level targets by 3 hours, 36 timesteps in advance
kkk = 6

X_SAMPLES = []
Y_SAMPLES = []
for mmm in range (5, len(RG1_data)-kkk):
    x_sample = [RG1_data[mmm-5], RG2_data[mmm-5], RG3_data[mmm-5], RG4_data[mmm-5], 
                RG1_data[mmm-4], RG2_data[mmm-4], RG3_data[mmm-4], RG4_data[mmm-4], 
                RG1_data[mmm-3], RG2_data[mmm-3], RG3_data[mmm-3], RG4_data[mmm-3], 
                RG1_data[mmm-2], RG2_data[mmm-2], RG3_data[mmm-2], RG4_data[mmm-2], 
                RG1_data[mmm-1], RG2_data[mmm-1], RG3_data[mmm-1], RG4_data[mmm-1], 
                RG1_data[mmm], RG2_data[mmm], RG3_data[mmm], RG4_data[mmm], 
                J13_data[mmm], J12_data[mmm], J4_data[mmm], J8_data[mmm], 
                ]
    y_sample = [T2_data[mmm+kkk], T3_data[mmm+kkk], T4_data[mmm+kkk], T6_data[mmm+kkk]]
    
    
    X_SAMPLES.append(x_sample)
    Y_SAMPLES.append(y_sample)
    



# Split the data into training and testing sets, with a 3:1 ratio, randomly
X_train, X_test, Y_train, Y_test = train_test_split(X_SAMPLES, Y_SAMPLES, test_size=0.25, random_state=42)



# Dataset collection
dataset_('X_train', X_train)
dataset_('Y_train', Y_train)

dataset_('X_test', X_test)
dataset_('Y_test', Y_test)






