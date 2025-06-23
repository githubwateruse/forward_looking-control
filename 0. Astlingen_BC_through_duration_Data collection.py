# -*- coding: utf-8 -*-
"""
Created on Wed May 14 01:21:24 2025

@author: sp825
"""


""" Original data collection per timestep: rainfall data and tanklevel data """
import os
# import pandas as pf
import numpy as np
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages

from datetime import datetime
import xlsxwriter

start_time = datetime.now()


control_time_step = 300 


swmm_inp = "C:/xiaoxin backup/end-to-end control/Astlingen simulation/Astlingen_SWMM_BC.inp"



RG1_RAINFALL, RG2_RAINFALL, RG3_RAINFALL, RG4_RAINFALL = [], [], [], []
T2_DEPTH, T3_DEPTH, T4_DEPTH, T6_DEPTH = [], [], [], []
J13_DEPTH, J12_DEPTH, J4_DEPTH, J8_DEPTH = [], [], [], []

C18_FLOW, C13_FLOW, C7_FLOW, C10_FLOW = [], [], [], []


with Simulation(swmm_inp) as sim:
    # sim.start_time = datetime(2000,1,1,0,0,0)
    # sim.end_time = datetime(2000,12,31,23,55,0)
    sim.step_advance(control_time_step)
    
    
    # init node object, link object, and subcatchment object
    subcatchment_object = Subcatchments(sim)
    SC01 = subcatchment_object["SC01"]
    SC02 = subcatchment_object["SC02"]
    SC03 = subcatchment_object["SC03"]
    SC04 = subcatchment_object["SC04"]
    SC05 = subcatchment_object["SC05"]
    SC06 = subcatchment_object["SC06"]
    SC07 = subcatchment_object["SC07"]
    SC08 = subcatchment_object["SC08"]
    SC09 = subcatchment_object["SC09"]
    SC010 = subcatchment_object["SC010"]
    
    node_object = Nodes(sim)  # init node object
    T1 = node_object["T1"]
    T2 = node_object["T2"]
    T3 = node_object["T3"]
    T4 = node_object["T4"]
    T5 = node_object["T5"]
    T6 = node_object["T6"]
    CSO7 = node_object["CSO7"]
    CSO8 = node_object["CSO8"]
    CSO9 = node_object["CSO9"]
    CSO10 = node_object["CSO10"]
    J1 = node_object["J1"]
    J2 = node_object["J2"]
    J3 = node_object["J3"]
    J4 = node_object["J4"]
    J5 = node_object["J5"]
    J6 = node_object["J6"]
    J7 = node_object["J7"]
    J8 = node_object["J8"]
    J9 = node_object["J9"]
    J10 = node_object["J10"]
    J11 = node_object["J11"]
    J12 = node_object["J12"]
    J13 = node_object["J13"]
    J14 = node_object["J14"]
    J15 = node_object["J15"]
    J16 = node_object["J16"]
    J17 = node_object["J17"]
    J18 = node_object["J18"]
    J19 = node_object["J19"]
    Out_to_WWTP = node_object["Out_to_WWTP"]
    
    link_object = Links(sim)  # init link object
    C14 = link_object["C14"]
    V1 = link_object["V1"]
    V2 = link_object["V2"]
    V3 = link_object["V3"]
    V4 = link_object["V4"]
    V5 = link_object["V5"]
    V6 = link_object["V6"]
    
    C18 = link_object["C18"]
    C13 = link_object["C13"]
    C7 = link_object["C7"]
    C10 = link_object["C10"]
    
    raingage_object = RainGages(sim) # init rain gages object
    RG1 = raingage_object["RG1"]
    RG2 = raingage_object["RG2"]
    RG3 = raingage_object["RG3"]
    RG4 = raingage_object["RG4"]
    
    aaa = 0
    print("start time", sim.start_time)
    print("end time", sim.end_time)
    
    
    for step in sim:

        
        
        RG1_RAINFALL.append(RG1.rainfall)
        RG2_RAINFALL.append(RG2.rainfall)
        RG3_RAINFALL.append(RG3.rainfall)
        RG4_RAINFALL.append(RG4.rainfall)
        
        J13_DEPTH.append(J13.depth)
        J12_DEPTH.append(J12.depth)
        J4_DEPTH.append(J4.depth)
        J8_DEPTH.append(J8.depth)
        
        C18_FLOW.append(C18.flow)
        C13_FLOW.append(C13.flow)
        C7_FLOW.append(C7.flow)
        C10_FLOW.append(C10.flow)
        
        T2_DEPTH.append(T2.depth)
        T3_DEPTH.append(T3.depth)
        T4_DEPTH.append(T4.depth)
        T6_DEPTH.append(T6.depth)
        
        
        
        
        
        pass
    

    sim.close()







# data collection
workbook_params = xlsxwriter.Workbook('rainfall_tanklevel_datasets/' + 'data_collection' +'.xlsx')
worksheet_params = workbook_params.add_worksheet('sheet1')

worksheet_params.write_column(0,0, RG1_RAINFALL)
worksheet_params.write_column(0,1, RG2_RAINFALL)
worksheet_params.write_column(0,2, RG3_RAINFALL)
worksheet_params.write_column(0,3, RG4_RAINFALL)


worksheet_params.write_column(0,5, J13_DEPTH)
worksheet_params.write_column(0,6, J12_DEPTH)
worksheet_params.write_column(0,7, J4_DEPTH)
worksheet_params.write_column(0,8, J8_DEPTH)


worksheet_params.write_column(0,10, T2_DEPTH)
worksheet_params.write_column(0,11, T3_DEPTH)
worksheet_params.write_column(0,12, T4_DEPTH)
worksheet_params.write_column(0,13, T6_DEPTH)


workbook_params.close()











end_time = datetime.now()
print("calculation time", end_time - start_time)





