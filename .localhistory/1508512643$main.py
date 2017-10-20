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

detect_final_result = main_function("1B3B2F")
print("The list of change points: {}".format(detect_final_result[1][0]))
print("The list of anomaly points: {}".format(detect_final_result[1][1]))

def fitFunc(x, b, c, d):
    return b*x[0] + c*x[1] + d*x[2]

##################### Prediction State #################################
raw_data = detect_final_result[0].value.values
anomaly_list = detect_final_result[1][1]
if (len(anomaly_list) >0):
    for j in anomaly_list:
        np.delete(raw_data,j)

plt.plot(np.arange(0, len(raw_data),raw_data))
plt.show()
change_points = detect_final_result[1][0]
data_test_range = []
data_test_value = []
for i in np.arange(1,4):
    change_point = change_points[-i]
    if (i == 1):
        data_train_size = int((len(raw_data) - change_point)*0.8);
        data_test_range = np.arange(data_train_size, len(raw_data) - change_point - data_train_size);
        data_test_value = raw_data[change_point+ data_train_size:]
    else:
        data_train_size = change_points[-i+1] - change_points[-i]

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




last_change_point = change_points[-1]
twoth_change_point = change_points[-2]
threeth_change_point = change_points[-3]

X = np.arange(0,1900 - last_change_point)
Y = raw_data[last_change_point:1900]

X_two = np.arange(0,last_change_point - twoth_change_point)
Y_two = raw_data[twoth_change_point :last_change_point]*(raw_data[last_change_point]/raw_data[twoth_change_point])

X_three = np.arange(0,twoth_change_point - threeth_change_point)
Y_three = raw_data[threeth_change_point : twoth_change_point]*(raw_data[last_change_point]/raw_data[threeth_change_point])

df = pd.DataFrame()
df = df.assign(X=pd.Series(X).values)
df = df.assign(Y=pd.Series(Y).values)

plt.plot(X,Y)
plt.plot(X_two,Y_two)
plt.plot(X_three,Y_three)
plt.show()

#linear_model_smf = smf.ols(formula='Y ~ X', data=df).fit()

#X_new = pd.DataFrame({'X': np.arange(0,len(raw_data) - last_change_point)})
#Y_new = linear_model_smf.predict(X_new)

#plt.plot(X,Y_new)

#print("Params: {}".format(linear_model_smf.params))
#print("R-squared: {}".format(linear_model_smf.rsquared))
########################### Online building model ############################
ridge_1 = Ridge(alpha=10)
ridge_1.fit(X.reshape(-1,1),Y)
print("Ridge model:{} + {}".format(pretty_print_linear(ridge_1.coef_),ridge_1.intercept_))
ridge_2 = Ridge(alpha=10)
ridge_2.fit(X_two.reshape(-1,1),Y_two)
print("Ridge model:{} + {}".format(pretty_print_linear(ridge_2.coef_),ridge_2.intercept_))
ridge_3 = Ridge(alpha=10)
ridge_3.fit(X_three.reshape(-1,1),Y_three)
print("Ridge model:{} + {}".format(pretty_print_linear(ridge_3.coef_),ridge_3.intercept_))

############################# Start Prediction ###############################

#print("Start prediction for point {}th:".format(len(raw_data) - 100))
X_test= np.arange(len(X), len(X)+100)
print(X_test)
Y_test = np.array([ridge_1.predict(X_test.reshape(-1,1)), ridge_2.predict(X_test.reshape(-1,1)), ridge_3.predict(X_test.reshape(-1,1))])
Y_train = raw_data[1900:2000]

plt.plot(raw_data)
plt.plot(np.arange(1900, 2000),Y_train)
plt.show()


fitParams, fitCovariances = curve_fit(fitFunc,Y_test, Y_train,bounds=(0, [1., 1., 1.]))
print(' fit coefficients:{}'.format(fitParams))

plt.plot(np.arange(1900, 2000),Y_train, label='Raw Data')
plt.plot(np.arange(1900, 2000),Y_test[0], label='Predict 1')
plt.plot(np.arange(1900, 2000),Y_test[1], label='Predict 2')
plt.plot(np.arange(1900, 2000),Y_test[2], label='Predict 3')
plt.plot(np.arange(1900, 2000),Y_test[0]*fitParams[0] + Y_test[1]*fitParams[1] + Y_test[2]*fitParams[2], label='Predict combined')
plt.legend()
plt.show()

