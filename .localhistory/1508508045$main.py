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

##################### Prediction State #################################
raw_data = detect_final_result[0].value.values
change_points = detect_final_result[1][0]
last_change_point = change_points[-1]
twoth_change_point = change_points[-2]
threeth_change_point = change_points[-3]

X = np.arange(0,len(raw_data) - 100 - last_change_point)
Y = raw_data[last_change_point:len(raw_data) - 100]

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

linear_model_smf = smf.ols(formula='Y ~ X', data=df).fit()

X_new = pd.DataFrame({'X': np.arange(0,len(raw_data) - last_change_point)})
Y_new = linear_model_smf.predict(X_new)

#plt.plot(X,Y_new)

print("Params: {}".format(linear_model_smf.params))
print("R-squared: {}".format(linear_model_smf.rsquared))
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
def fitFunc(x, b, c, d):
    return b*x[0] + c*x[1] + d*x[2]
#print("Start prediction for point {}th:".format(len(raw_data) - 100))
X_test= np.arange(len(raw_data) - 100 - last_change_point, len(raw_data) - last_change_point)
Y_test = np.array([ridge_1.predict(X_test.reshape(-1,1)), ridge_2.predict(X_test.reshape(-1,1)), ridge_3.predict(X_test.reshape(-1,1))])
Y_train = raw_data[len(raw_data) - 100:]
print((Y_test))
print((Y_train))
fitParams, fitCovariances = curve_fit(fitFunc,Y_test, Y_train,bounds=(0, [1., 1., 1.]))
print(' fit coefficients:{}'.format(fitParams))
#ridge_conf = Ridge(alpha=1,fit_intercept =False, normalize=True)
#ridge_conf.fit(Y_test,Y_train)
#print("Correlating model:{} + {}".format(pretty_print_linear(ridge_conf.coef_),ridge_conf.intercept_))



