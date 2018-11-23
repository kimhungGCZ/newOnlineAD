import numpy as np
import pandas as pd
import math as math
import scripts.common_functions as cmfunc
import sklearn.neighbors as nb
from sklearn.neighbors import DistanceMetric
import threading
import time
import os
import json
import statsmodels.formula.api as smf
from datetime import datetime


import detection_engine as engine
import scripts.obtain_data as data_engine

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

from sklearn.linear_model import Ridge
from sympy.solvers import solve
from sympy import Symbol
from random import randint

x = Symbol('x')


import warnings

warnings.simplefilter('ignore')

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def pretty_print_linear(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

def main_function(DATA_FILE, AN_per, CP_per, max_AN, max_CP):
    #DATA_FILE = "2004DF"
    # if not os.path.exists('graph/' + DATA_FILE):
    #     os.makedirs('graph/' + DATA_FILE)
    #DATA_FILE = "example " + str(randint(1, 1000000)) + ".csv"
    #DATA_FILE = "example " + str(randint(1, 1000000))
    #DATA_FILE =
    start = time.time()
    start_getting_data = time.time()
    raw_dataframe = data_engine.getGCZDataFrame(DATA_FILE, AN_per, CP_per)
    end_getting_data = time.time()
    print("Getting Data Time: {}".format(start_getting_data - end_getting_data))
    raw_dta = raw_dataframe.value.values
    print("Length Data {}:".format(len(raw_dta)))

    # dao ham bac 1
    der = cmfunc.change_after_k_seconds_with_abs(raw_dta, k=1)
    # dao ham bac 2
    sec_der = cmfunc.change_after_k_seconds_with_abs(der, k=1)

    median_sec_der = np.median(sec_der)
    mean_sec_der = np.mean(sec_der)
    std_sec_der = np.std(sec_der)
    mad_sec_der = np.median([np.abs(i - mean_sec_der) for i in sec_der])

    # breakpoint_candidates = list(map(lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
    #     enumerate(sec_der)))
    breakpoint_candidates = list(map(lambda x: (x[1] - mad_sec_der)  if (x[1] - median_sec_der) > 0 else 0,enumerate(sec_der)))

    #breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

    breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)
    breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)

    # data_value = raw_dataframe.value;
    # split_data = np.array_split(data_value, int(len(data_value) / 10))
    # std_whole_dataset = np.std(data_value)
    # print(0.03 * std_whole_dataset)
    # # sdev_array = [second_dereviate(i) for i in split_data]
    # sdev_array = [np.mean(cmfunc.change_after_k_seconds(running_mean(i.values,2))) for i in split_data]
    #
    # #bins = np.array([0.0 - float("inf"), 0.0 - 0.03 * std_whole_dataset, 0.0 + 0.03 * std_whole_dataset, float("inf")])
    # bins = np.array([0.0 - float("inf"), 0.0 - 0.5 , 0.0 + 0.05 , float("inf")])
    #
    # inds = np.digitize(sdev_array, bins)
    # plt.plot(data_value)
    # plt.show()



    ############### To debug specific combination:############################
    final_index = 0
    alpha = 0.05
    print("Decay Value: %f" % alpha)
    new_data = raw_dataframe.copy()
    new_data = new_data.assign(anomaly_score=pd.Series(breakpoint_candidates).values)
    start_main_al = time.time()
    detect_final_result = engine.online_anomaly_detection(new_data, raw_dataframe, alpha, DATA_FILE, max_AN, max_CP)


    end_main_al = time.time()
    print("Execution time: {}".format(end_main_al - start_main_al))
    #engine.generate_tsing_data_format(DATA_FILE, raw_dataframe, detect_final_result[3])
    # print("The list of change points: {}".format(detect_final_result[0]))
    # print("The list of anomaly points: {}".format(detect_final_result[1]))
    return detect_final_result


if __name__== "__main__":
    base_name = "real_"
    data_array = []  # real
    #data_array = ["example 320387", "example 346500", "example 533964","example 645266"] # SAW
    #data_array = ["example 624622", "example 798717", "example 513024"] # SIN
    #data_array = ["example 348800", "example 387713", "example 692083", "example 961480", "example 989638"] # SQUARE
    data_array = ["test_1111"] # real
    #data_array = ["real 624622"] # real
    #if i in [1,3,4,5,8,12,14,21,22,23,24,25,42]:
    # for i in range(1,50):
    #     if i not in [7,10,20]:
    #         data_array.append(base_name + str(i))


    #data_array = ["real_42"]
    for data in data_array:

        #AL_coup = [[0.01,85],[0.05,80],[0.1,75],[0.15,70],[0.2,65]]
        #CP_coup = [0.02, 0.05, 0.1, 0.15]
        CP_coup = [0.01]
        #CP_coup = [0.01, 0.02, 0.05, 0.1, 0.15]
        #AL_coup = [[0.1,75],[0.15,70],[0.2,65]]
        AL_coup = [[0.01,60]]
        for CP_value in CP_coup:
            for run_value in AL_coup:
                runn_flag = 0
                flag_stop = 0
                data = ("test_"+str(run_value[0])+"_" +str(CP_value)).replace(".","")
                max_AN = 0
                max_CP = 0
                while runn_flag <= 0:
                    detect_final_result = main_function(data, run_value[0], CP_value, max_AN, max_CP)
                    print(detect_final_result)
                    runn_flag = runn_flag + 1
                    # if detect_final_result[0] > max_AN and detect_final_result[1] > max_CP:
                    #     max_AN = detect_final_result[0]
                    #     max_CP = detect_final_result[1]
                    # if detect_final_result[0] >= run_value[1] and detect_final_result[1] >= 65:
                    #     runn_flag = 1
                    max_AN = detect_final_result[1]
                    max_CP = detect_final_result[2]
                    print("MAX AN: {}, MAX AP: {}".format(max_AN, max_CP))
                # while flag_stop == 0:
                #     detect_final_result = main_function(data, run_value[0], 0.05)
                #     print(detect_final_result)
                #     print("Result: {}, MAX AN: {}, MAX AP: {}".format(detect_final_result,max_AN, max_CP))
                #     if detect_final_result[0] >= max_AN and detect_final_result[1] >= max_CP:
                #         flag_stop = 1
                # print("Final Result: {}, {}".format(detect_final_result[0], detect_final_result[1]))
                print("****************************************************************************************")




