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

def main_function(DATA_FILE, K_value):
    #DATA_FILE = "2004DF"
    # if not os.path.exists('graph/' + DATA_FILE):
    #     os.makedirs('graph/' + DATA_FILE)
    #DATA_FILE = "example " + str(randint(1, 1000000)) + ".csv"
    #DATA_FILE = "example " + str(randint(1, 1000000))
    #DATA_FILE =
    start = time.time()
    start_getting_data = time.time()
    raw_dataframe = data_engine.getGCZDataFrame(DATA_FILE)
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

    breakpoint_candidates = list(map(lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
        enumerate(sec_der)))
    breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

    breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)
    breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)

    data_value = raw_dataframe.value;
    split_data = np.array_split(data_value, int(len(data_value) / 10))
    std_whole_dataset = np.std(data_value)
    print(0.03 * std_whole_dataset)
    # sdev_array = [second_dereviate(i) for i in split_data]
    sdev_array = [np.mean(cmfunc.change_after_k_seconds(running_mean(i.values,2))) for i in split_data]

    #bins = np.array([0.0 - float("inf"), 0.0 - 0.03 * std_whole_dataset, 0.0 + 0.03 * std_whole_dataset, float("inf")])
    bins = np.array([0.0 - float("inf"), 0.0 - 0.5 , 0.0 + 0.05 , float("inf")])

    inds = np.digitize(sdev_array, bins)
    # plt.plot(data_value)
    # plt.show()



    ############### To debug specific combination:############################
    final_index = 0
    alpha = 0.05
    print("Decay Value: %f" % alpha)
    new_data = raw_dataframe.copy()
    new_data = new_data.assign(anomaly_score=pd.Series(breakpoint_candidates).values)
    start_main_al = time.time()
    detect_final_result = engine.online_anomaly_detection(new_data, raw_dataframe, alpha, DATA_FILE, K_value)
    end_main_al = time.time()
    print("Execution time: {}".format(end_main_al - start_main_al))

    # print("The list of change points: {}".format(detect_final_result[0]))
    # print("The list of anomaly points: {}".format(detect_final_result[1]))
    return detect_final_result

if __name__== "__main__":
    base_name = "real_"
    data_array = []  # real
    #data_array = ["example 320387", "example 346500", "example 533964","example 645266"] # SAW
    #data_array = ["example 624622", "example 798717", "example 513024"] # SIN
    #data_array = ["example 348800", "example 387713", "example 692083", "example 961480", "example 989638"] # SQUARE
    #data_array = ["real_8"] # real
    # for i in range(1,50):
    #     if i not in [7,10,20]:
    #         data_array.append(base_name + str(i))
    AL_coup = [[0.01, 85], [0.05, 80], [0.1, 75], [0.15, 70], [0.2, 65]]
    CP_coup = [0.01, 0.02, 0.05, 0.1, 0.2]
    # CP_coup = [0.2]
    # CP_coup = [0.01, 0.02, 0.05, 0.1, 0.15]
    # AL_coup = [[0.1,75],[0.15,70],[0.2,65]]
    # AL_coup = [[0.01,60]]
    for CP_value in CP_coup:
        for run_value in AL_coup:
            data_name = ("test_" + str(run_value[0]) + "_" + str(CP_value)).replace(".", "")
            data_array.append(data_name)


    K_value_array = np.arange(5,100,5)
    for data in data_array:
        detect_result_in_K = []
        for K_value in K_value_array:
            print("############################# START AT DATASET: {}, K = {} ##########################################".format(data, K_value))
            detect_final_result = main_function(data, K_value)
            detect_result_in_K.append(detect_final_result)
        detect_result_in_K = np.array(detect_result_in_K)
        detect_result_in_K = detect_result_in_K[detect_result_in_K[:,2].argsort()][-1]
        try:
            df_final_result = pd.read_csv(os.path.normpath(
                'D:/Google Drive/13. These cifre/Data Cleaning/workspace/knn_new_syn/' + 'final_value' + '.csv'))

            df_final_result = df_final_result.append({'dataset': detect_result_in_K[0],
                                                      'bf_f_score_anomaly': detect_result_in_K[1],
                                                      'af_f_score_anomaly': detect_result_in_K[2],
                                                      'bf_f_score_changepoint': detect_result_in_K[3],
                                                      'af_f_score_changepoint': detect_result_in_K[4],
                                                      'nb_anomaly_point': detect_result_in_K[5],
                                                      'nb_change_point': detect_result_in_K[6],
                                                      'query': detect_result_in_K[7]}, ignore_index=True)
            df_final_result.to_csv(os.path.normpath(
                'D:/Google Drive/13. These cifre/Data Cleaning/workspace/knn_new_syn/' + 'final_value' + '.csv'), index=False);
        except FileNotFoundError:

            df_final_result = pd.DataFrame([[detect_result_in_K[0],
                                             detect_result_in_K[1],
                                             detect_result_in_K[2],
                                             detect_result_in_K[3],
                                             detect_result_in_K[4],
                                             detect_result_in_K[5],
                                             detect_result_in_K[6],
                                             detect_result_in_K[7],
                                             ]],
                                           columns=['dataset', 'bf_f_score_anomaly', 'af_f_score_anomaly','bf_f_score_changepoint','af_f_score_changepoint','nb_anomaly_point', 'nb_change_point',
                                                    'query'])
            df_final_result.to_csv(os.path.normpath(
                'D:/Google Drive/13. These cifre/Data Cleaning/workspace/knn_new_syn/' + 'final_value' + '.csv'), index=False);




