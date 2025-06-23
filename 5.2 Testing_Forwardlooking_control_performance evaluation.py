# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:55:51 2025

@author: sp825
"""


# import some modules
import numpy as np
import multiprocessing as mp
import time

import os
from datetime import datetime
import math


import matplotlib.pyplot
import xlrd

from xlrd import xldate_as_datetime, xldate_as_tuple
import xlsxwriter
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages


from sklearn.neural_network import MLPRegressor
import joblib  

# Environmental programming module: SWMM simulation
# --------------------------------------------------------------------------------
class BasicEnv(object):
    def __init__(self, inp_file, testing_year):
        
        
        
        # initialize simulation
        self.input_file = inp_file
        self.sim = Simulation(self.input_file)  # read input file
       
        self.control_time_step = 300  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step
        
        self.ep_len = int((self.sim.end_time - self.sim.start_time).total_seconds()/self.control_time_step)
        
        
        # init node object, link object, and subcatchment object
        subcatchment_object = Subcatchments(self.sim)
        self.SC01 = subcatchment_object["SC01"]
        self.SC02 = subcatchment_object["SC02"]
        self.SC03 = subcatchment_object["SC03"]
        self.SC04 = subcatchment_object["SC04"]
        self.SC05 = subcatchment_object["SC05"]
        self.SC06 = subcatchment_object["SC06"]
        self.SC07 = subcatchment_object["SC07"]
        self.SC08 = subcatchment_object["SC08"]
        self.SC09 = subcatchment_object["SC09"]
        self.SC010 = subcatchment_object["SC010"]
        
        node_object = Nodes(self.sim)  # init node object
        self.T1 = node_object["T1"]
        self.T2 = node_object["T2"]
        self.T3 = node_object["T3"]
        self.T4 = node_object["T4"]
        self.T5 = node_object["T5"]
        self.T6 = node_object["T6"]
        self.CSO7 = node_object["CSO7"]
        self.CSO8 = node_object["CSO8"]
        self.CSO9 = node_object["CSO9"]
        self.CSO10 = node_object["CSO10"]
        self.J1 = node_object["J1"]
        self.J2 = node_object["J2"]
        self.J3 = node_object["J3"]
        self.J4 = node_object["J4"]
        self.J5 = node_object["J5"]
        self.J6 = node_object["J6"]
        self.J7 = node_object["J7"]
        self.J8 = node_object["J8"]
        self.J9 = node_object["J9"]
        self.J10 = node_object["J10"]
        self.J11 = node_object["J11"]
        self.J12 = node_object["J12"]
        self.J13 = node_object["J13"]
        self.J14 = node_object["J14"]
        self.J15 = node_object["J15"]
        self.J16 = node_object["J16"]
        self.J17 = node_object["J17"]
        self.J18 = node_object["J18"]
        self.J19 = node_object["J19"]
        self.Out_to_WWTP = node_object["Out_to_WWTP"]
        
        link_object = Links(self.sim)  # init link object
        self.C14 = link_object["C14"]
        self.V1 = link_object["V1"]
        self.V2 = link_object["V2"]
        self.V3 = link_object["V3"]
        self.V4 = link_object["V4"]
        self.V5 = link_object["V5"]
        self.V6 = link_object["V6"]
        
        raingage_object = RainGages(self.sim) # init rain gages object
        self.RG1 = raingage_object["RG1"]
        self.RG2 = raingage_object["RG2"]
        self.RG3 = raingage_object["RG3"]
        self.RG4 = raingage_object["RG4"]

        
        # the statistics of CSO volume of each node
        self.T1_CSO, self.T2_CSO, self.T3_CSO, self.T4_CSO, self.T5_CSO, self.T6_CSO = 0, 0, 0, 0, 0, 0
        self.CSO7_CSO, self.CSO8_CSO, self.CSO9_CSO, self.CSO10_CSO = 0, 0, 0, 0
        self.J1_CSO, self.J2_CSO, self.J3_CSO, self.J4_CSO, self.J5_CSO, self.J6_CSO = 0, 0, 0, 0, 0, 0
        self.J7_CSO, self.J8_CSO, self.J9_CSO, self.J10_CSO, self.J11_CSO, self.J12_CSO = 0, 0, 0, 0, 0, 0
        self.J13_CSO, self.J14_CSO, self.J15_CSO, self.J16_CSO, self.J17_CSO, self.J18_CSO, self.J19_CSO = 0, 0, 0, 0, 0, 0, 0
        
        
        
        
        self.tankdepth = np.asarray([self.T2.depth,  self.T3.depth,  self.T4.depth, self.T6.depth,])
        self.rainfalldata = np.asarray([self.RG1.rainfall,  self.RG2.rainfall,  self.RG3.rainfall, self.RG4.rainfall,])
        
                   
        
    def step(self, action):
        
        self.V2.target_setting = np.round(np.double(action[0]), decimals = 2)
        self.V3.target_setting = np.round(np.double(action[1]), decimals = 2)
        self.V4.target_setting = np.round(np.double(action[2]), decimals = 2)
        self.V6.target_setting = np.round(np.double(action[3]), decimals = 2)
        self.sim.__next__()
        
        
        current_time = self.sim.current_time
        
        
        self.nodedepth = np.asarray([self.J13.depth, self.J12.depth, self.J4.depth, self.J8.depth])
        self.tankdepth = np.asarray([self.T2.depth,  self.T3.depth,  self.T4.depth, self.T6.depth,])
        self.rainfalldata = np.asarray([self.RG1.rainfall,  self.RG2.rainfall,  self.RG3.rainfall, self.RG4.rainfall,])
        
        
                 
        cso_volume = (self.T1.statistics['flooding_volume'] - self.T1_CSO) + (self.T2.statistics['flooding_volume'] - self.T2_CSO) \
                    + (self.T3.statistics['flooding_volume'] - self.T3_CSO) + (self.T4.statistics['flooding_volume'] - self.T4_CSO) \
                    + (self.T5.statistics['flooding_volume'] - self.T5_CSO) + (self.T6.statistics['flooding_volume'] - self.T6_CSO) \
                    + (self.CSO7.statistics['flooding_volume'] - self.CSO7_CSO) + (self.CSO8.statistics['flooding_volume'] - self.CSO8_CSO) \
                    + (self.CSO9.statistics['flooding_volume'] - self.CSO9_CSO) + (self.CSO10.statistics['flooding_volume'] - self.CSO10_CSO) \
                     
        
        self.T1_CSO, self.T2_CSO = self.T1.statistics['flooding_volume'], self.T2.statistics['flooding_volume']
        self.T3_CSO, self.T4_CSO = self.T3.statistics['flooding_volume'], self.T4.statistics['flooding_volume']
        self.T5_CSO, self.T6_CSO = self.T5.statistics['flooding_volume'], self.T6.statistics['flooding_volume']
        
        self.CSO7_CSO, self.CSO8_CSO = self.CSO7.statistics['flooding_volume'], self.CSO8.statistics['flooding_volume']
        self.CSO9_CSO, self.CSO10_CSO = self.CSO9.statistics['flooding_volume'], self.CSO10.statistics['flooding_volume']
        
        
        if self.t < self.T-1:
            done = False
        else:
            done = True
        
        self.t += 1
        
        info = {}
        
        return self.nodedepth, self.tankdepth, self.rainfalldata, done, cso_volume, current_time
    
    def reset(self, testing_year):
        
        self.sim.close()
        
        self.sim = Simulation(self.input_file)
        
        self.sim.step_advance(self.control_time_step)  # set control time step
        
       
        self.sim.start_time = datetime(testing_year, 1, 1, 0,0,0)
        self.sim.end_time = datetime(testing_year, 12, 31, 23,55,0)
        
        # define start simulation time and end simulation time
        print("reset", "start_time", self.sim.start_time, "end_time", self.sim.end_time)
        self.ep_len = int((self.sim.end_time - self.sim.start_time).total_seconds()/self.control_time_step)
        
        # init node object, link object, and subcatchment object
        subcatchment_object = Subcatchments(self.sim)
        self.SC01 = subcatchment_object["SC01"]
        self.SC02 = subcatchment_object["SC02"]
        self.SC03 = subcatchment_object["SC03"]
        self.SC04 = subcatchment_object["SC04"]
        self.SC05 = subcatchment_object["SC05"]
        self.SC06 = subcatchment_object["SC06"]
        self.SC07 = subcatchment_object["SC07"]
        self.SC08 = subcatchment_object["SC08"]
        self.SC09 = subcatchment_object["SC09"]
        self.SC010 = subcatchment_object["SC010"]
        
        node_object = Nodes(self.sim)  # init node object
        self.T1 = node_object["T1"]
        self.T2 = node_object["T2"]
        self.T3 = node_object["T3"]
        self.T4 = node_object["T4"]
        self.T5 = node_object["T5"]
        self.T6 = node_object["T6"]
        self.CSO7 = node_object["CSO7"]
        self.CSO8 = node_object["CSO8"]
        self.CSO9 = node_object["CSO9"]
        self.CSO10 = node_object["CSO10"]
        self.J1 = node_object["J1"]
        self.J2 = node_object["J2"]
        self.J3 = node_object["J3"]
        self.J4 = node_object["J4"]
        self.J5 = node_object["J5"]
        self.J6 = node_object["J6"]
        self.J7 = node_object["J7"]
        self.J8 = node_object["J8"]
        self.J9 = node_object["J9"]
        self.J10 = node_object["J10"]
        self.J11 = node_object["J11"]
        self.J12 = node_object["J12"]
        self.J13 = node_object["J13"]
        self.J14 = node_object["J14"]
        self.J15 = node_object["J15"]
        self.J16 = node_object["J16"]
        self.J17 = node_object["J17"]
        self.J18 = node_object["J18"]
        self.J19 = node_object["J19"]
        self.Out_to_WWTP = node_object["Out_to_WWTP"]
        
        link_object = Links(self.sim)  # init link object
        self.C14 = link_object["C14"]
        self.V1 = link_object["V1"]
        self.V2 = link_object["V2"]
        self.V3 = link_object["V3"]
        self.V4 = link_object["V4"]
        self.V5 = link_object["V5"]
        self.V6 = link_object["V6"]
        
        raingage_object = RainGages(self.sim) # init rain gages object
        RG1 = raingage_object["RG1"]
        RG2 = raingage_object["RG2"]
        RG3 = raingage_object["RG3"]
        RG4 = raingage_object["RG4"]
        
        
        self.sim.start()
        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.control_time_step)
        self.t = 1
        
        
        self.nodedepth = np.asarray([self.J13.depth, self.J12.depth, self.J4.depth, self.J8.depth])
        self.tankdepth = np.asarray([self.T2.depth,  self.T3.depth,  self.T4.depth, self.T6.depth,])
        self.rainfalldata = np.asarray([self.RG1.rainfall,  self.RG2.rainfall,  self.RG3.rainfall, self.RG4.rainfall,])
        
        
        return self.nodedepth, self.tankdepth, self.rainfalldata, self.T
    
    def close(self):
        self.sim.report()
        self.sim.close()




model_filename = 'rainfall_action_datasets/rainfall_data_vs_action_30min_advance.pkl'

forward_looking_control_model = joblib.load(model_filename)

def FORWARD_LOOKING_CONTROL(testing_year):
    
    CSO_Volume_Per_Step = []
    Time_Per_Step = []
    
    
    
    """ Using historical rainfall data and tanklevel to guide decision-making """
    
    Historical_rainfalldata = []
    Historical_tankdepth = []
    Historical_nodedepth = []
    swmm_file = "C:/xiaoxin backup/end-to-end control/Astlingen simulation/Astlingen_SWMM_ES.inp"
    
 
    storm_env = BasicEnv(swmm_file, testing_year)  

    Nodedepth, Tankdepth, Rainfalldata, EP_LEN = storm_env.reset(testing_year)
    
    Historical_rainfalldata.append(Rainfalldata)
    Historical_tankdepth.append(Tankdepth)
    Historical_nodedepth.append(Nodedepth)
    
    done = False
    for t in range (EP_LEN):
        
        
        # Since there’s no historical data at the beginning, use the default BC decision,
        # but make sure to update the system state as well.
        
        if t <= 11:
            # V2 = 0.2366, V3 = 0.6508, V4 = 0.3523, V6 = 0.4303
            Action = [0.2366, 0.6508, 0.3523, 0.4303]
            Orifice_setting = Action
            
        else:
            policy_input = np.array([[Historical_rainfalldata[t-11][0], Historical_rainfalldata[t-11][1], Historical_rainfalldata[t-11][2], Historical_rainfalldata[t-11][3],
                                     Historical_rainfalldata[t-10][0], Historical_rainfalldata[t-10][1], Historical_rainfalldata[t-0][2], Historical_rainfalldata[t-10][3],
                                     Historical_rainfalldata[t-9][0], Historical_rainfalldata[t-9][1], Historical_rainfalldata[t-9][2], Historical_rainfalldata[t-9][3],
                                     Historical_rainfalldata[t-8][0], Historical_rainfalldata[t-8][1], Historical_rainfalldata[t-8][2], Historical_rainfalldata[t-8][3],
                                     Historical_rainfalldata[t-7][0], Historical_rainfalldata[t-7][1], Historical_rainfalldata[t-7][2], Historical_rainfalldata[t-7][3],
                                     Historical_rainfalldata[t-6][0], Historical_rainfalldata[t-6][1], Historical_rainfalldata[t-6][2], Historical_rainfalldata[t-6][3],
                                     Historical_nodedepth[t-6][0], Historical_nodedepth[t-6][1], Historical_nodedepth[t-6][2], Historical_nodedepth[t-6][3],]])
            
            
            Action =  forward_looking_control_model.predict(policy_input)
            Orifice_setting = Action[0]
        
        
       
        
        Nodedepth, Tankdepth, Rainfalldata, done, CSO_volume, current_time = storm_env.step(Orifice_setting)
        
        
        Historical_rainfalldata.append(Rainfalldata)
        Historical_tankdepth.append(Tankdepth)
        Historical_nodedepth.append(Nodedepth)
        
        CSO_Volume_Per_Step.append(CSO_volume)
        Time_Per_Step.append(current_time)
        
        done_bool = float(done)
        

        
        if done:
            storm_env.close()
            break
            
    return CSO_Volume_Per_Step, Time_Per_Step


if __name__ == "__main__":
    Train_time_start = time.time()
    
    
    
    workbook_curve = xlsxwriter.Workbook('Performance comparison/Forward_looking control policy_Overall' +'.xlsx')
    worksheet_curve = workbook_curve.add_worksheet('sheet1')
    
    
    TESTING_YEARS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    
    for mmm in range (len(TESTING_YEARS)):
        testing_year = TESTING_YEARS[mmm]
        print("testing_year", testing_year)
        
        
        CSO_Volume_Per_Step, Time_Per_Step = FORWARD_LOOKING_CONTROL(testing_year)
        
        
        workbook_year = xlsxwriter.Workbook('Performance comparison/Forward_looking control policy_Overall_year ' + str(testing_year) +'.xlsx')
        worksheet_year = workbook_year.add_worksheet('sheet1')
        
        
        worksheet_year.write_column(0,0, Time_Per_Step)
        worksheet_year.write_column(0,1, CSO_Volume_Per_Step)
        
        workbook_year.close()
        
        
        logfile = open("C:/xiaoxin backup/end-to-end control/Astlingen simulation/Astlingen_SWMM_ES.rpt","r",-1,"utf-8")
    
        kws_start = ["Node Flooding Summary",]
        kws_end = ["Storage Volume Summary",]
        lines = logfile.readlines()#获取每行的数据
        # print(len(lines))
        for m in range (len(lines)):
            if (any (kw in lines[m] for kw in kws_start)):
                start_lines = m + 10
            if (any (kw in lines[m] for kw in kws_end)):
                end_lines = m - 4
                
        # print(lines[Report_start_lines])
        AAA = []
        BBB = []
        BBB_creek = []
        BBB_river = []
        for n in range (start_lines, end_lines+1):
            Res = lines[n].split('    ')
            # print(Res[0], np.round(float(Res[-2]), decimals = 3))
            AAA.append(Res[0])
            cso_volume = np.round(float(Res[-2]), decimals = 3)
            BBB.append(cso_volume)
            
            if Res[0] == '  CSO7' or Res[0] == '  CSO9' or Res[0] == '  T6':
                BBB_creek.append(cso_volume)
            else:
                BBB_river.append(cso_volume)
    
        print("total", np.sum(BBB))
        # print("creek", np.sum(BBB_creek))
        # print("river", np.sum(BBB_river))
        
        
        
        logfile.close()
        
        
        
        
        
        worksheet_curve.write_column(mmm,0, [np.sum(BBB)])
        worksheet_curve.write_column(mmm,1, [np.sum(BBB_creek)])
        worksheet_curve.write_column(mmm,2, [np.sum(BBB_river)])
        
        
        os.remove("C:/xiaoxin backup/end-to-end control/Astlingen simulation/Astlingen_SWMM_ES.rpt")
        os.remove("C:/xiaoxin backup/end-to-end control/Astlingen simulation/Astlingen_SWMM_ES.out")
        print("File removed successfully")
    
       
        
    workbook_curve.close()
    Train_time_end = time.time()
    print("total time consumption", Train_time_end - Train_time_start)
    
