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

import detection_engine as engine
import scripts.obtain_data as data_engine

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

from sklearn.linear_model import Ridge



import warnings

warnings.simplefilter('ignore')

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
    if not os.path.exists('graph/' + DATA_FILE):
        os.makedirs('graph/' + DATA_FILE)
    start = time.time()
    start_getting_data = time.time()
    raw_dataframe = data_engine.getGCZDataFrame(DATA_FILE)
    end_getting_data = time.time()
    print("Getting Data Time: {}".format(start_getting_data - end_getting_data))
    raw_dta = raw_dataframe.value.values
    print("Length Data {}:".format(len(raw_dta)))

    # dao ham bac 1
    #der = cmfunc.change_after_k_seconds(raw_dta.value, k=1)
    # dao ham bac 2
    sec_der = cmfunc.change_after_k_seconds_with_abs(raw_dta, k=1)

    median_sec_der = np.median(sec_der)
    std_sec_der = np.std(sec_der)

    breakpoint_candidates = list(map(lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
        enumerate(sec_der)))
    breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

    breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=4)
    final_f = []
    final_combination = []

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

    end = time.time()
    print("Total time: {}".format(end - start))
    return [raw_dataframe, detect_final_result]

detect_final_result = main_function("2004DF")
print("The list of change points: {}".format(detect_final_result[1][0]))
print("The list of anomaly points: {}".format(detect_final_result[1][1]))

def fitFunc_2(x, b, c):
    return b*x[0] + c*x[1]
def fitFunc_3(x, b, c, d):
    return b*x[0] + c*x[1] + d*x[2]
##################### Prediction State #################################
raw_data = detect_final_result[0].value.values
anomaly_list = detect_final_result[1][1]
anomaly_list_array = np.array(anomaly_list)
raw_data = raw_data[0:len(raw_data)-10]
raw_data = np.delete(raw_data, anomaly_list)
change_points = [0] + detect_final_result[1][0]
# Update changepoint list after remove anomaly point.
change_points = [i - len(anomaly_list_array[anomaly_list_array < i]) for i in change_points]
data_test_range = []
data_test_value = []
data_predicting_test = []
data_predicting_plot = []
plt.plot(raw_data)
if len(change_points) < 3:
    start_range = np.arange(1,len(change_points)+1)
else:
    start_range = np.arange(1,4)

for i in start_range:
   change_point = change_points[-i]
   if (i == 1):
       data_train_size = int((len(raw_data) - change_point)*0.2);
       data_test_range = np.arange(data_train_size, len(raw_data) - change_point);
       data_test_value = raw_data[change_point+ data_test_range]
       plt.plot(change_point+ data_test_range, data_test_value)
       plt.show()
   else:
       data_train_size = change_points[-i+1] - change_points[-i] - 1

   data_train_X = np.arange(0, data_train_size)
   if (i == 1):
       data_train_Y = raw_data[change_point:change_point+data_train_size]
   else:
       data_train_Y = raw_data[change_point:change_point+data_train_size]*(raw_data[change_points[-1]]/raw_data[change_points[-i]])
    
   plt.plot(data_train_X, data_train_Y)
   plt.show()
   ####  BUIDING MODEL ####
   ridge_1 = Ridge(alpha=10)
   ridge_1.fit(data_train_X.reshape(-1,1),data_train_Y)
   print("Ridge model:{} + {}".format(pretty_print_linear(ridge_1.coef_),ridge_1.intercept_))

   data_predicting_test.append(ridge_1.predict(data_test_range.reshape(-1,1)))
   data_predicting_plot.append(ridge_1.predict(np.arange(0, len(raw_data) - change_points[-1]).reshape(-1,1)))

if len(change_points) >= 3:
    fitParams, fitCovariances = curve_fit(fitFunc_3, data_predicting_test, data_test_value, bounds=(0, [1., 1., 1.]))
elif len(change_points) == 2:
    fitParams, fitCovariances = curve_fit(fitFunc_2, data_predicting_test, data_test_value, bounds=(0, [1., 1.]))
else:
    fitParams, fitCovariances = data_predicting_test[0]
print('fit coefficients:{}'.format(fitParams))
print('Standard deviation error: {}'.format(np.sqrt(np.diag(fitCovariances))))

plt.plot(np.arange(0, len(raw_data) - change_points[-1]), fitFunc_3(data_predicting_plot, *fitParams), 'g--', label='fit-with-bounds')
plt.plot(np.arange(0, len(raw_data) - change_points[-1]), raw_data[change_points[-1]:], 'r', label='data')
plt.show()
