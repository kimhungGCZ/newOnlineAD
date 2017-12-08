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

def main_function(DATA_FILE):
    #DATA_FILE = "2004DF"
    # if not os.path.exists('graph/' + DATA_FILE):
    #     os.makedirs('graph/' + DATA_FILE)
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
    detect_final_result = engine.online_anomaly_detection(new_data, raw_dataframe, alpha, DATA_FILE)
    end_main_al = time.time()
    print("Execution time: {}".format(end_main_al - start_main_al))

    print("The list of change points: {}".format(detect_final_result[0]))
    print("The list of anomaly points: {}".format(detect_final_result[1]))
    return detect_final_result

    """
        def fitFunc_1(x, a, b):
        return a * x + b

    def fitFunc_2(x, b, c):
        return b * x[0] + c * x[1]

    def fitFunc_3(x, b, c, d):
        return b * x[0] + c * x[1] + d * x[2]

    ##################### Prediction State #################################
    #raw_data = raw_dta
    anomaly_list = detect_final_result[1]
    anomaly_list_array = np.array(anomaly_list)
    #raw_data = raw_data[0:len(raw_data) - 10]
    raw_dataframe = raw_dataframe.drop(raw_dataframe.index[np.arange(len(raw_dataframe)-10, len(raw_dataframe))])
    #raw_data = np.delete(raw_data, anomaly_list)
    raw_dataframe = raw_dataframe.drop(raw_dataframe.index[anomaly_list])
    change_points = [0] + detect_final_result[0]
    # Update changepoint list after remove anomaly point.
    change_points = [i - len(anomaly_list_array[anomaly_list_array < i]) for i in change_points]
    raw_data = raw_dataframe.value.values
    time_stamp = raw_dataframe.timestamp.values
    data_test_range = []
    data_test_value = []
    data_predicting_test = []
    data_predicting_plot = []
    model_params = []
    data_predicting_test_last10 = []
    test_size = 10
    raw_data_last_10 = raw_data[-test_size:]
    raw_data = raw_data[:len(raw_data) - test_size]

    #plt.plot(raw_data)
    if len(change_points) < 3:
        start_range = np.arange(1, len(change_points) + 1)
    else:
        start_range = np.arange(1, 4)

    for i in start_range:
        change_point = change_points[-i]
        if (i == 1):
            data_train_size = int((len(raw_data) - change_point) * 0.9);
            data_test_range = np.arange(data_train_size, len(raw_data) - change_point);
            data_test_value = raw_data[change_point + data_test_range]
            # plt.plot(change_point + data_test_range, data_test_value)
            # plt.show()
        else:
            data_train_size = change_points[-i + 1] - change_points[-i] - 1

        data_train_X = np.arange(0, data_train_size)
        if (i == 1):
            data_train_Y = raw_data[change_point:change_point + data_train_size]
        else:
            data_train_Y = raw_data[change_point:change_point + data_train_size] * (
            raw_data[change_points[-1]] / raw_data[change_points[-i]])

        # plt.plot(data_train_X, data_train_Y)
        # plt.show()

        ####  BUIDING MODEL ####
        ridge_1 = Ridge(alpha=10)
        ridge_1.fit(data_train_X.reshape(-1, 1), data_train_Y)
        print("Ridge model {}:{} + {}".format(str(i), pretty_print_linear(ridge_1.coef_), ridge_1.intercept_))
        model_params.append([ridge_1.coef_[0],ridge_1.intercept_])
        data_predicting_test.append(ridge_1.predict(data_test_range.reshape(-1, 1)))
        data_predicting_test_last10.append(
            ridge_1.predict(np.arange(max(data_test_range), max(data_test_range) + test_size).reshape(-1, 1)))
        #plt.plot(data_test_range, ridge_1.predict(data_test_range.reshape(-1, 1)), label="Predict Model " + str(i));

        data_predicting_plot.append(ridge_1.predict(np.arange(0, len(raw_data) - change_points[-1]).reshape(-1, 1)))

    # plt.plot(data_test_range, data_test_value, label="Real Data")
    # plt.legend();
    # plt.show()
    if len(change_points) >= 3:
        fitParams, fitCovariances = curve_fit(fitFunc_3, data_predicting_test, data_test_value,
                                              bounds=(0, [1., 1., 1.]))
    elif len(change_points) == 2:
        fitParams, fitCovariances = curve_fit(fitFunc_2, data_predicting_test, data_test_value, bounds=(0, [1., 1.]))
    else:
        fitParams, fitCovariances = data_predicting_test[0]
    print('fit coefficients:{}'.format(fitParams))
    print('Standard deviation error: {}'.format(np.sqrt(np.diag(fitCovariances))))

    plt.plot(data_test_range, fitFunc_3(data_predicting_test, *fitParams), 'g--', label='fit-with-bounds')
    # plt.plot(np.arange(0, len(raw_data) - change_points[-1]), raw_data[change_points[-1]:], 'r', label='data')
    # plt.plot(change_point+ data_test_range, data_test_value, label='test data')
    plt.plot(np.arange(max(data_test_range), max(data_test_range) + test_size),
             fitFunc_3(data_predicting_test_last10, *fitParams), label='Test Next Points')
    plt.plot(np.arange(max(data_test_range), max(data_test_range) + test_size),
             fitFunc_1(np.arange(max(data_test_range), max(data_test_range) + test_size), *model_params[0]), label='1st model prediction')
    plt.plot(np.arange(max(data_test_range), max(data_test_range) + test_size),
             fitFunc_1(np.arange(max(data_test_range), max(data_test_range) + test_size), *model_params[1]), label='2st model prediction')
    plt.plot(np.arange(max(data_test_range), max(data_test_range) + test_size),
             fitFunc_1(np.arange(max(data_test_range), max(data_test_range) + test_size), *model_params[2]), label='3st model prediction')
    plt.plot(np.arange(max(data_test_range), max(data_test_range) + test_size), raw_data_last_10,
             label='Real Next Points')
    plt.legend();
    plt.show()

    end = time.time()
    print("Total time: {}".format(end - start))
    print("Model Params: {}".format(model_params))
    time_model_1_to_0 = (solve(fitFunc_1(x, *model_params[0]), x))
    time_model_2_to_0 =(solve(fitFunc_1(x, *model_params[1]), x))
    time_model_3_to_0 =(solve(fitFunc_1(x, *model_params[2]), x))
    print("Estimate point to reach 0 at model 1: {}".format(time_model_1_to_0))
    print("Estimate point to reach 0 at model 2: {}".format(time_model_2_to_0)) 
    print("Estimate point to reach 0 at model 3: {}".format(time_model_3_to_0))

    finish_prediction_time = datetime.strptime(time_stamp[-1], "%Y-%m-%d %H:%M:%S")
    start_prediction_time = datetime.strptime(time_stamp[change_points[-1]], "%Y-%m-%d %H:%M:%S")
    difference_days = (finish_prediction_time - start_prediction_time).days
    difference_distances = int((len(raw_data) - change_points[-1]))
    time_per_day_eachDtaPoint_1 = int(time_model_1_to_0[0]*difference_days/difference_distances) - difference_days if (int(time_model_1_to_0[0]*difference_days/difference_distances) - difference_days) > 0 else 0
    time_per_day_eachDtaPoint_2 = int(time_model_2_to_0[0]*difference_days/difference_distances) - difference_days if (int(time_model_2_to_0[0]*difference_days/difference_distances) - difference_days) >0 else 0
    time_per_day_eachDtaPoint_3 = (int(time_model_3_to_0[0]*difference_days/difference_distances) - difference_days) if int(time_model_3_to_0[0]*difference_days/difference_distances) - difference_days > 0 else 0

    print("Time to empty by days: {}".format(fitFunc_3([time_per_day_eachDtaPoint_1,time_per_day_eachDtaPoint_2,time_per_day_eachDtaPoint_3],*fitParams)))


    return [time_per_day_eachDtaPoint_1,time_per_day_eachDtaPoint_2,time_per_day_eachDtaPoint_3]
    """

detect_final_result = main_function("2004DF")



