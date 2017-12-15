from fnmatch import filter

import numpy as np
import pandas as pd
import math as math
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
from itertools import islice

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def find_inverneghboor_of_point_blocking(alpha, index_ano_list, result_dta, Z):
    start_time_S = time.time();
    in_X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(np.max(index_ano_list)+10)))
    in_tree = nb.KDTree(in_X, leaf_size=200)
    for index_ano in index_ano_list:
        limit_size = int(1 / alpha)
        inverse_neighboor = set()
        inverse_neighboor_temp = set()
        anomaly_point = in_X[index_ano]
        flag_stop = 0
        flag_round = 2
        len_inverse_neighboor = 0
        while flag_stop <= limit_size:
            # time.sleep(0.05)
            # flag_stop +=1
            len_start = len_inverse_neighboor
            dist, ind = in_tree.query([anomaly_point], k=flag_round)
            for index_dist, i in enumerate(ind[0]):
                if (index_dist, i) not in inverse_neighboor:
                    if len_inverse_neighboor != 0:
                        if i not in inverse_neighboor_temp:
                            in_dist, in_ind = in_tree.query([in_X[i]], k=flag_round)
                            if ((index_ano in in_ind[0])):  # or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                                inverse_neighboor.add(
                                    (index_dist, i))  # np.append(inverse_neighboor, [index_dist, i], axis=0)
                                len_inverse_neighboor += 1
                                inverse_neighboor_temp.add(i)
                    else:
                        in_dist, in_ind = in_tree.query([in_X[i]], k=flag_round)
                        if ((index_ano in in_ind[0])):  # or (check_in_array(in_ind[0], inverse_neighboor) == 1):
                            inverse_neighboor.add(
                                (index_dist, i))  # np.append(inverse_neighboor, [index_dist, i], axis=0)
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

        nomaly_neighboor = np.array(list(inverse_neighboor), dtype=np.int32)
        for NN_pair in nomaly_neighboor:
            Z[NN_pair[1]] = Z[NN_pair[1]] + (1 - result_dta['anomaly_score'][index_ano]) - NN_pair[0] * alpha if (1 -
                                                                                                                  result_dta[
                                                                                                                      'anomaly_score'][
                                                                                                                      index_ano]) - \
                                                                                                                 NN_pair[
                                                                                                                     0] * alpha > 0 else \
                Z[NN_pair[1]]
            # print("Find invert neighbor {}th Time: {} ".format(index_ano, time.time()-start_time_S));
    return time.time() - start_time_S


def calculate_Y_value(alpha, anomaly_point, limit_size, median_sec_der, potential_anomaly, raw_dta, result_dta,
                      std_sec_der, tree, X, Y):

    # start_size = 100
    # increase_size = 0
    # # increase_size = start_size;
    # query_point = anomaly_point + start_size
    # # flag_running = False
    # # while increase_size>1:
    # #     dist, ind = tree.query([X[int(query_point)]], k=start_size + 1)
    # #     dist_ano, ind_ano = tree.query([X[int(anomaly_point)]], k=start_size + 1)
    # #     if anomaly_point in ind[0]:
    # #         increase_size = increase_size;
    # #         query_point = query_point + increase_size;
    # #     else:
    # #         increase_size = increase_size/2;
    # #         query_point = query_point - increase_size;
    #
    # while increase_size == 0:
    #     dist, ind = tree.query([X[int(query_point)]], k=start_size)
    #     if np.max(ind[0]) <= query_point:
    #         increase_size = 1
    #     else:
    #         query_point = query_point + 1
    # print("Finding time {}".format(time.time() - sssss))
    #
    #
    #
    # query_array = raw_dta.value[int(anomaly_point):int(query_point)]
    # std_checked_dataset = np.std(query_array)
    # split_data = np.array_split(query_array, int(len(query_array) / 10))
    # sdev_array = [np.mean(cmfunc.change_after_k_seconds(i)) for i in split_data]
    #
    # bins = np.array([0.0 - float("inf"), 0.0 - 0.03 * std_checked_dataset, 0.0 + 0.03 * std_checked_dataset, float("inf")])
    # inds = np.digitize(sdev_array, bins)
    # plt.hist(inds)
    # plt.show()
    # max_access = 0
    # max_bin = 0
    # for bin_index in np.arange(1,4):
    #     if (inds == bin_index).sum() > max_access:
    #         max_bin = bin_index
    sssss = time.time();

    flag_starting = False
    flag_running = False;

    if anomaly_point - 1 not in potential_anomaly and anomaly_point - 2 not in potential_anomaly:
        if anomaly_point - 1 not in potential_anomaly:
            anomaly_neighboor_detect = np.array(
                cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 1, limit_size),
                dtype=np.int32)
        else:
            anomaly_neighboor_detect = np.array(
                cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 2, limit_size),
                dtype=np.int32)
        if len(set(anomaly_neighboor_detect[:, 1]).intersection(potential_anomaly)) == 0:
            anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                         dtype=np.int32)
            potential_anomaly.extend([x[1] for x in anomaly_neighboor])
            flag_starting = True
        else:
            consider_point = np.max(
                [i for i in list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
                 if
                 i not in potential_anomaly])
            if (raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                consider_point] - median_sec_der - std_sec_der > 0):
                anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                             dtype=np.int32)
                potential_anomaly.extend((x[1] for x in anomaly_neighboor))
                flag_starting = True
            else:
                result_dta.anomaly_score[anomaly_point] = 0
    else:
        consider_point = max(set(np.arange(anomaly_point)).difference(set(potential_anomaly)))
        if (abs(raw_dta.value.values[anomaly_point] - raw_dta.value.values[
            consider_point]) - median_sec_der - std_sec_der >= 0):
            anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                         dtype=np.int32)
            potential_anomaly.extend([x[1] for x in anomaly_neighboor])
            flag_starting = True
        else:
            result_dta.anomaly_score[anomaly_point] = 0

    if flag_starting == True:
        anomaly_neighboor_detect = np.array(
            cmfunc.find_inverneghboor_of_point_2(tree, X, anomaly_point, limit_size),
            dtype=np.int32)
        print("Spreading length of {}: {}".format(anomaly_point, len(anomaly_neighboor_detect)))
        x = anomaly_neighboor_detect[:, 1]
        y = result_dta['value'][anomaly_neighboor_detect[:, 1]]
        m, b = np.polyfit(x, y, 1)
        print("Coffecient M: {}".format(m))




    if flag_running == True:
        if anomaly_point - 1 not in potential_anomaly and anomaly_point - 2 not in potential_anomaly:
            if anomaly_point - 1 not in potential_anomaly:
                anomaly_neighboor_detect = np.array(
                    cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 1, limit_size),
                    dtype=np.int32)
            else:
                anomaly_neighboor_detect = np.array(
                    cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 2, limit_size),
                    dtype=np.int32)
            if len(set(anomaly_neighboor_detect[:, 1]).intersection(potential_anomaly)) == 0:
                anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                             dtype=np.int32)
                potential_anomaly.extend([x[1] for x in anomaly_neighboor])
                for NN_pair in anomaly_neighboor:
                    Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                        result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
            else:
                consider_point = np.max(
                    [i for i in list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
                     if
                     i not in potential_anomaly])
                if (raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                    consider_point] - median_sec_der - std_sec_der > 0):
                    anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                                 dtype=np.int32)
                    potential_anomaly.extend((x[1] for x in anomaly_neighboor))
                    for NN_pair in anomaly_neighboor:
                        Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[
                                                                                                         0] * alpha if \
                            result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
                else:
                    result_dta.anomaly_score[anomaly_point] = 0
        else:
            consider_point = max(set(np.arange(anomaly_point)).difference(set(potential_anomaly)))
            if (abs(raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                consider_point]) - median_sec_der - std_sec_der >= 0):
                anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                             dtype=np.int32)
                potential_anomaly.extend([x[1] for x in anomaly_neighboor])
                for NN_pair in anomaly_neighboor:
                    Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                        result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
            else:
                result_dta.anomaly_score[anomaly_point] = 0

    return sssss - time.time()


def online_anomaly_detection(result_dta, raw_dta, alpha, DATA_FILE):

    median_sec_der = np.mean(result_dta['value'])
    std_sec_der = np.std(result_dta['value'])

    dta_full = result_dta

    dta_full.value.index = result_dta.timestamp

    std_anomaly_set = np.std(result_dta['anomaly_score'])
    np.argsort(result_dta['anomaly_score'])

    # Get 5% anomaly point
    # anomaly_index =
    # np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set])

    limit_size = int(0.05* len(result_dta['anomaly_score']))
    # Y is the anomaly spreding and Z is the normal spreading.
    Y = np.zeros(len(result_dta['anomaly_score']))
    Z = np.zeros(len(result_dta['anomaly_score']))
    X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(len(result_dta.values))))

    tree = nb.KDTree(X, leaf_size=200, metric='euclidean')
    potential_anomaly = []
    executor_y = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )
    start_time_calculate_Y = time.time()

    ######################################################
    # Calculate Y
    # Asyncronous function to calculate Y value
    async def calculate_Y_value_big(executor):
        loop = asyncio.get_event_loop()
        blocking_tasks = [
            loop.run_in_executor(executor, calculate_Y_value, alpha, anomaly_point, limit_size, median_sec_der,
                                 potential_anomaly, raw_dta, result_dta, std_sec_der, tree, X, Y)
            for anomaly_point in sorted(anomaly_index)
        ]
        completed, pending = await asyncio.wait(blocking_tasks)
        results = [t.result() for t in completed]
        print("The sum times of calculating Y: {}".format(sum(results)))

    # Starting calculating function
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    try:
        event_loop.run_until_complete(
            calculate_Y_value_big(executor_y)
        )
    finally:
        event_loop.close()
        print("Finish threading calculation of Y")

    # Calculate final score
    result_dta.anomaly_score = result_dta.anomaly_score + Y

    print("Calculating Y Time for {} elements: {}".format(len(anomaly_index), time.time() - start_time_calculate_Y))

    #######################################################
    ########## STARTING CALCULATING Z ####################

    ssss = time.time()
    # normal_index = [i for i, value in enumerate(result_dta['anomaly_score']) if
    #                filter_function_z(result_dta['anomaly_score'],i,limit_size/2,data_length) > 0 and value <= np.percentile(result_dta['anomaly_score'], 20)]
    anomaly_score = result_dta['anomaly_score']
    normal_index = [i for i, value in enumerate(anomaly_score) if
                    value <= 0 and anomaly_score[tree.query([X[i]], k=int(limit_size / 2))[1][0]].values.max() > 0]
    print("Chossing time for {}: {}".format(len(normal_index), time.time() - ssss))
    normal_points_array = chunks(sorted(normal_index), 30)

    async def calculate_z_value(executor):
        loop = asyncio.get_event_loop()
        blocking_tasks = [
            loop.run_in_executor(executor, find_inverneghboor_of_point_blocking, alpha, normal_points,
                                 result_dta, Z)
            for normal_points in normal_points_array
        ]
        completed, pending = await asyncio.wait(blocking_tasks)
        results = [t.result() for t in completed]
        print("The sum times of calculating Z: {}".format(sum(results)))

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    try:
        event_loop.run_until_complete(
            calculate_z_value(executor)
        )
    finally:
        event_loop.close()

    result_dta.anomaly_score = result_dta.anomaly_score - Z

    end_time_calculate_Z = time.time()
    print("Calculating Z Time: {}".format(time.time() - ssss))

    final_score = list(map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score))
    final_score = (final_score - np.min(final_score)) / (np.max(final_score) - np.min(final_score))

    # Calculating Change point.
    start_time_calculate_changepoint = time.time()

    ### Find potential anomaly point
    std_final_point = np.std(final_score)

    anomaly_set = [i for i, v in enumerate(final_score) if v > 0]

    # The algorithm to seperate anomaly point and change point.
    X = list(map(lambda x: [x, x], np.arange(len(result_dta.values))))
    newX = list(np.array(X)[anomaly_set])
    newtree = nb.KDTree(X, leaf_size=50)

    anomaly_group_set = []
    new_small_x = 0
    sliding_index = 1
    for index_value, new_small_x in enumerate(anomaly_set):
        anomaly_neighboor = np.array(
            cmfunc.find_inverneghboor_of_point_1(newtree, X, new_small_x, anomaly_set, limit_size),
            dtype=np.int32)
        tmp_array = list(map(lambda x: x[1], anomaly_neighboor))
        if index_value > 0:
            common_array = list(set(tmp_array).intersection(anomaly_group_set[index_value - sliding_index]))
            if len(common_array) != 0:
                union_array = list(set(tmp_array).union(anomaly_group_set[index_value - sliding_index]))
                anomaly_group_set[index_value - sliding_index] = np.append(
                    anomaly_group_set[index_value - sliding_index],
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

        list_of_anomaly = [int(j) for i in anomaly_group_set for j in i]
        example_data = [i for i in (
            list(raw_dta.value.values[list(z for z in range(int(min(detect_pattern) - 3), int(min(detect_pattern))) if
                                           z not in list_of_anomaly)]) + list(
                raw_dta.value.values[
                    list(z for z in range(int(max(detect_pattern) + 1), int(max(detect_pattern) + 4)) if
                         z not in list_of_anomaly and z < len(raw_dta.value.values))]))]

        in_std_with_Anomaly = np.std(
            example_data + list(raw_dta.value.values[int(min(detect_pattern)): int(max(detect_pattern) + 1)]))
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

    return detect_final_result
    # return [detect_final_result,chartmess]
