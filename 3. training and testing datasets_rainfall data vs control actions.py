# -*- coding: utf-8 -*-
"""
Created on Wed May 14 07:04:56 2025

@author: sp825
"""
""" 
This .py file is used for training and testing dataset split (More accurately, just focus on Y_Samples)

Timestep is 5 minutes

X_sample: Data from RG1, RG2, RG3, and RG4 over the past 30 minutes, 
          as well as the current depth of J13, J12, J4, and J8
(hence, the file "rainfall_action_X_test.xlsx" is the same as the file "rainfall_tanklevel_X_test.xlsx")



Y_sample: Actions of V2, V3, V4, and V6 in the future 30 minutes
(The data from the file "rainfall_tanklevel_Y_test.xlsx" serves as the input to the control policies, based on which the file "rainfall_action_Y_test.xlsx" can be generated.)

""" 

import numpy as np
import xlsxwriter
import xlrd
from datetime import datetime

start_time = datetime.now()

# Functions implementing a centralized control policy, mapping tank levels to corresponding control actions
# =================================================================================
def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p


def build_net(len_state, len_action):
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    
    s0, p0 = linear(len_state, 30)
    s1, p1 = linear(30, 30)
    s2, p2 = linear(30, len_action)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


def relu_func(inX):
    return np.maximum(0, inX)
    
    


# Extracting centralized control policy parameters
# =================================================================================
policy_path = "C:/xiaoxin backup/end-to-end control/Astlingen simulation/rainfall_action_datasets/Centralized control policy_NN parameters.xlsx"

LEN_STATE = 4 
LEN_ACTION = 4


net_shapes, net_params = build_net(LEN_STATE, LEN_ACTION)


workbook = xlrd.open_workbook(policy_path)
data_sheet = workbook.sheet_by_index(0) 
rowNum = data_sheet.nrows
colNum = data_sheet.ncols
net_params = np.array(data_sheet.col_values(0))

Centralized_control_params = params_reshape(net_shapes, net_params)




    
def get_action(x):
    
    params = Centralized_control_params
    
    x = x[np.newaxis, :]
    x = relu_func(x.dot(params[0]) + params[1])
    x = relu_func(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
   
    return 0.5 * np.tanh(x)[0] + 0.5             # for continuous action




# Extracting the inputs (i.e., tanklevels) for the centralized control policy
# =================================================================================

#  file_name = "Y_train" or file_name = "Y_test"
file_name = "Y_test"  



TANKLEVELS = []

data_path_name = "C:/xiaoxin backup/end-to-end control/Astlingen simulation/rainfall_action_datasets/rainfall_tanklevel_" + file_name + ".xlsx"

workbook = xlrd.open_workbook(data_path_name)
data_sheet = workbook.sheet_by_index(0) 
rowNum = data_sheet.nrows
colNum = data_sheet.ncols

for mmm in range (rowNum):
    tanklevels =  data_sheet.row_values(mmm)
    TANKLEVELS.append(tanklevels)


Centralized_policy_input = np.array(TANKLEVELS)
    



# Generating the outputs (i.e., control actions) based on the centralized control policy and inputs
# =================================================================================

Control_Actions = []
for nnn in range (len(Centralized_policy_input)):
    actions = get_action(Centralized_policy_input[nnn])
    Control_Actions.append(actions)
    
    




# Y_samples collection
# =================================================================================     
workbook_params = xlsxwriter.Workbook('rainfall_action_datasets/' + 'rainfall_action_'+ file_name +'.xlsx')
worksheet_params = workbook_params.add_worksheet('sheet1')

data_ = Control_Actions
for nnn in range (len(data_)):
    worksheet_params.write_row(nnn, 0, data_[nnn])

workbook_params.close()

end_time = datetime.now()
print("calculation time", end_time - start_time)
