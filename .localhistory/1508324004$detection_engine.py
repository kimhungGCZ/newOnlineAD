from fnmatch import filter

import numpy as np
import pandas as pd
import math as math
from scipy.stats import norm
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import scripts.common_functions as cmfunc
import sklearn.neighbors as nb
from sklearn.neighbors import DistanceMetric
import trollius
import warnings
from numpy import mean, absolute
import time
import asyncio  
import inspect
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

warnings.simplefilter('ignore')


# groud_trust = [[350, 832], [732, 733, 734, 735, 736, 755, 762, 773, 774,
# 795]]


# groud_trust = [[581,1536],[435, 460, 471, 557, 558, 559, 560, 561, 562, 563,
# 564, 570, 571, 572, 573, 574, 1174, 1175, 1383, 1418, 1423]]

def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)



async def calculate_Y_value(alpha, anomaly_point, limit_size, median_sec_der, potential_anomaly, raw_dta, result_dta, std_sec_der, tree, X, Y):
    if anomaly_point - 1 not in potential_anomaly:
        anomaly_neighboor_detect = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 1, limit_size),
            dtype=np.int32)
        if len(set(anomaly_neighboor_detect[:, 1]).intersection(potential_anomaly)) == 0:
            anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                         dtype=np.int32)
            potential_anomaly.extend([x[1] for x in anomaly_neighboor])
            for NN_pair in anomaly_neighboor:
                Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                    result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
        else:
            consider_point = np.max([i for i in list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
                 if
                 i not in potential_anomaly])
            if (raw_dta.value.values[anomaly_point] - raw_dta.value.values[consider_point] - median_sec_der - std_sec_der > 0):
                anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                             dtype=np.int32)
                potential_anomaly.extend((x[1] for x in anomaly_neighboor))
                for NN_pair in anomaly_neighboor:
                    Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                        result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
            else:
                result_dta.anomaly_score[anomaly_point] = 0
    else:
        temp_X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(anomaly_point)))
        # dt=DistanceMetric.get_metric('pyfunc',func=mydist)
        temp_tree = nb.KDTree(temp_X, leaf_size=20)
        anomaly_neighboor_detect = np.array(cmfunc.find_inverneghboor_of_point(temp_tree, temp_X, anomaly_point - 1, limit_size),
            dtype=np.int32)
        consider_point = np.max([i for i in list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1]))) if i not in potential_anomaly])
    
        if (abs(raw_dta.value.values[anomaly_point] - raw_dta.value.values[consider_point]) - median_sec_der - std_sec_der > 0):
            anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                         dtype=np.int32)
            potential_anomaly.extend([x[1] for x in anomaly_neighboor])
            for NN_pair in anomaly_neighboor:
                Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                    result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
        else:
            result_dta.anomaly_score[anomaly_point] = 0

def find_invert(index_ano):
    inverse_neighboor = set()
    inverse_neighboor_temp = set()
    anomaly_point = X[index_ano]
    flag_stop = 0
    flag_round = 2
    len_inverse_neighboor = 0
    while flag_stop <= int(limit_size/2):
        #time.sleep(0.05)
        #flag_stop +=1
        len_start = len_inverse_neighboor
        dist, ind = tree.query([anomaly_point], k=flag_round)
        for index_dist, i in enumerate(ind[0]):
            if (index_dist, i) not in inverse_neighboor:
                if len_inverse_neighboor != 0:
                    if i not in inverse_neighboor_temp:
                        in_dist, in_ind = tree.query([X[i]], k=flag_round)
                        if ((index_ano in in_ind[0])) :#or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                            inverse_neighboor.add( (index_dist, i))  # np.append(inverse_neighboor, [index_dist, i], axis=0)
                            len_inverse_neighboor += 1
                            inverse_neighboor_temp.add(i)
                else:
                    in_dist, in_ind = tree.query([X[i]], k=flag_round)
                    if ((index_ano in in_ind[0])) :#or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                        inverse_neighboor.add((index_dist, i))  # np.append(inverse_neighboor, [index_dist, i], axis=0)
                        len_inverse_neighboor += 1
                        inverse_neighboor_temp.add(i)
        len_stop = len_inverse_neighboor
        if len_start == len_stop:
            flag_stop += 1
            flag_round += 1
        else:
            # Reset flag_stop and flag_round when found
            flag_stop = 0
        if len_inverse_neighboor > limit_size:
            break
    return inverse_neighboor, inverse_neighboor_temp

def online_anomaly_detection(result_dta, raw_dta, alpha, DATA_FILE):

    # dao ham bac 2
    sec_der = cmfunc.change_after_k_seconds_with_abs(raw_dta.value, k=1)

    median_sec_der = np.median(sec_der)
    std_sec_der = np.std(sec_der)

    breakpoint_candidates = list(map(lambda x: (x[1] - median_sec_der) - np.abs(std_sec_der) if (x[1] - median_sec_der) - np.abs(std_sec_der) > 0 else 0,
        enumerate(sec_der)))
    breakpoint_candidates = (breakpoint_candidates - np.min(breakpoint_candidates)) / (np.max(breakpoint_candidates) - np.min(breakpoint_candidates))

    breakpoint_candidates = np.insert(breakpoint_candidates, 0, 0)

    dta_full = result_dta

    dta_full.value.index = result_dta.timestamp

    std_anomaly_set = np.std(result_dta['anomaly_score'])
    np.argsort(result_dta['anomaly_score'])

    # Get 5% anomaly point
    # anomaly_index =
    # np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set])

    limit_size = int(1 / alpha)
    # Y is the anomaly spreding and Z is the normal spreading.
    Y = np.zeros(len(result_dta['anomaly_score']))
    Z = np.zeros(len(result_dta['anomaly_score']))
    X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(len(result_dta.values))))
    # dt=DistanceMetric.get_metric('pyfunc',func=mydist)
    tree = nb.KDTree(X, leaf_size=50)
    potential_anomaly = []
    def find_inverneghboor_of_point_blocking(index_ano):
        start_time_S = time.time();
        inverse_neighboor, inverse_neighboor_temp = find_invert(index_ano)

        nomaly_neighboor = np.array(list(inverse_neighboor), dtype=np.int32)
        for NN_pair in nomaly_neighboor:
            Z[NN_pair[1]] = Z[NN_pair[1]] + (1 - result_dta['anomaly_score'][index_ano]) - NN_pair[0] * alpha if (1 - result_dta['anomaly_score'][index_ano]) - \
                                                                                                                    NN_pair[0] * alpha > 0 else \
                Z[NN_pair[1]]
        print("Find invert neighbor {}th Time: {} - return value {}".format(index_ano, time.time()-start_time_S, inverse_neighboor_temp));
        return time.time()-start_time_S

    async def calculate_y_value(executor):
        #print(normal_point)
        #await asyncio.sleep(1)  await loop.run_in_executor(ProcessPoolExecutor(), sleep, delay)
        s= time.time()
        loop = asyncio.get_event_loop()
        blocking_tasks = [
            loop.run_in_executor(executor, find_inverneghboor_of_point_blocking, normal_point)
            for normal_point in sorted(normal_index)
        ]
        completed, pending = await asyncio.wait(blocking_tasks)
        results = [t.result() for t in completed]
        print(sum(results))

    start_time_calculate_Y = time.time()
    # Calculate Y
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )

    tasks = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for anomaly_point in anomaly_index:
        #calculate_Y_value(alpha, anomaly_point, limit_size, median_sec_der, potential_anomaly, raw_dta, result_dta, std_sec_der, tree, X, Y)
        tasks.append(asyncio.ensure_future(calculate_Y_value(alpha, anomaly_point, limit_size, median_sec_der, potential_anomaly, raw_dta, result_dta, std_sec_der, tree, X, Y)))

    loop.run_until_complete(asyncio.wait(tasks))
#    loop.close()
    backup_draw = result_dta.copy()

    # Calculate final score
    result_dta.anomaly_score = result_dta.anomaly_score + Y

    end_time_calculate_Y = time.time()
    print("Calculating Y Time: {}".format(start_time_calculate_Y - end_time_calculate_Y))

    start_time_calculate_Z = time.time()
    # Find normal point
    # normal_index =
    # np.array(np.argsort(result_dta['anomaly_score']))[:int((0.4 *
    # len(result_dta['anomaly_score'])))]
    normal_index = [i for i, value in enumerate(result_dta['anomaly_score']) if
                    value <= np.percentile(result_dta['anomaly_score'], 20)]

    normal_index = np.random.choice(normal_index, int(len(normal_index) * 0.2), replace=False)


    async def calculate_z_value(executor):
        #print(normal_point)
        #await asyncio.sleep(1)  await loop.run_in_executor(ProcessPoolExecutor(), sleep, delay)
        s= time.time()
        loop = asyncio.get_event_loop()
        blocking_tasks = [
            loop.run_in_executor(executor, find_inverneghboor_of_point_blocking, normal_point)
            for normal_point in sorted(normal_index)
        ]
        completed, pending = await asyncio.wait(blocking_tasks)
        results = [t.result() for t in completed]
        print(sum(results))

    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(
            calculate_z_value(executor)
        )
    finally:
        event_loop.close()
        

    result_dta.anomaly_score = result_dta.anomaly_score - Z

    end_time_calculate_Z = time.time()
    print("Calculating Z Time: {}".format(start_time_calculate_Z - end_time_calculate_Z))

    final_score = list(map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score))
    final_score = (final_score - np.min(final_score)) / (np.max(final_score) - np.min(final_score))

    # Calculating Change point.
    start_time_calculate_changepoint = time.time()

    ### Find potential anomaly point
    std_final_point = np.std(final_score)
    # anomaly_set = [i for i, v in enumerate(final_score) if v > 3 *
    # std_final_point]
    anomaly_set = [i for i, v in enumerate(final_score) if v > 0]

    # The algorithm to seperate anomaly point and change point.
    X = list(map(lambda x: [x, x], np.arange(len(result_dta.values))))
    newX = list(np.array(X)[anomaly_set])
    newtree = nb.KDTree(X, leaf_size=50)

    anomaly_group_set = []
    new_small_x = 0
    sliding_index = 1
    for index_value, new_small_x in enumerate(anomaly_set):
        anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point_1(newtree, X, new_small_x, anomaly_set, limit_size),
            dtype=np.int32)
        tmp_array = list(map(lambda x: x[1], anomaly_neighboor))
        if index_value > 0:
            common_array = list(set(tmp_array).intersection(anomaly_group_set[index_value - sliding_index]))
            # anomaly_group_set = np.concatenate((anomaly_group_set,
            # tmp_array))
            if len(common_array) != 0:
                union_array = list(set(tmp_array).union(anomaly_group_set[index_value - sliding_index]))
                anomaly_group_set[index_value - sliding_index] = np.append(anomaly_group_set[index_value - sliding_index],
                    list(set(tmp_array).difference(anomaly_group_set[index_value - sliding_index])))
                sliding_index = sliding_index + 1
            else:
                anomaly_group_set.append(np.sort(tmp_array))
        else:
            anomaly_group_set.append(np.sort(tmp_array))

    new_array = [tuple(row) for row in anomaly_group_set]
    uniques = new_array
    std_example_data = []
    std_example_outer = []
    detect_final_result = [[], []]
    for detect_pattern in uniques:
        # rest_anomaly_set = [i for i in anomaly_set if i not in
        # list(detect_pattern)]
        list_of_anomaly = [int(j) for i in anomaly_group_set for j in i]
        example_data = [i for i in (list(raw_dta.value.values[list(z for z in range(int(min(detect_pattern) - 3), int(min(detect_pattern))) if
                                           z not in list_of_anomaly)]) + list(raw_dta.value.values[list(z for z in range(int(max(detect_pattern) + 1), int(max(detect_pattern) + 4)) if
                    z not in list_of_anomaly and z < len(raw_dta.value.values))]))]

        in_std_with_Anomaly = np.std(example_data + list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern) + 1)]))
        std_example_data.append(in_std_with_Anomaly)

        example_data_iner = list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern)) + 1])

        in_std_with_NonAnomaly = np.std(example_data)
        if (in_std_with_Anomaly > 1.5 * in_std_with_NonAnomaly):
            detect_final_result[1].extend(np.array(detect_pattern, dtype=np.int))
        else:
            detect_final_result[0].append(int(np.min(detect_pattern)))
        std_example_outer.append(in_std_with_NonAnomaly)
    final_changepoint_set = detect_final_result[0]
    data_file = DATA_FILE

    end_time_calculate_changepoint = time.time()
    print("Calculating Change Point Time: {}".format(start_time_calculate_changepoint - end_time_calculate_changepoint))
    chartmess = cmfunc.plot_data_all(DATA_FILE,
                         [[list(range(0, len(raw_dta.value))), raw_dta.value],
                          [detect_final_result[0], raw_dta.value[detect_final_result[0]]],
                          [detect_final_result[1], raw_dta.value[detect_final_result[1]]]],
                         ['lines', 'markers', 'markers'], [None, 'circle', 'circle', 'x', 'x'],
                         ['Raw data', "Detected Change Point",
                          "Detected Anomaly Point"])
    return [detect_final_result]
    #return [detect_final_result,chartmess]