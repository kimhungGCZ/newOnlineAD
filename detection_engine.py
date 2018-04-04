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
# import trollius
import warnings
from numpy import mean, absolute
import time
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import datetime
import os

from saxpy.saxpy.znorm import znorm
from saxpy.saxpy.sax import ts_to_string
from saxpy.saxpy.alphabet import cuts_for_asize

from scripts.detect_peaks import detect_peaks

cuts_for_asize(3)

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
        yield l[i:i + n]


def find_inverneghboor_of_point_blocking(alpha, index_ano_list, result_dta, Z):
    start_time_S = time.time();
    in_X = list(map(lambda x: [x, result_dta.values[x][1]], np.arange(np.max(index_ano_list) + 10)))
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


def calculate_Y_value(alpha, anomaly_point, limit_size, median_sec_der, dict_neighbor, raw_dta, result_dta,
                      std_sec_der, tree, X, Y):

    sssss = time.time();
    if anomaly_point in dict_neighbor:
        anomaly_neighboor = dict_neighbor[anomaly_point]
        # potential_anomaly.extend([x[1] for x in anomaly_neighboor])
        for NN_pair in anomaly_neighboor:
            Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
                result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]

    ############################# THE CODE IS WORK WELL WIH VERSION 1.05, COMMENT TO TEST WITH VERSION 1.06 ######################

    # if flag_running == True:
    #     if anomaly_point - 1 not in potential_anomaly and anomaly_point - 2 not in potential_anomaly:
    #         if anomaly_point - 1 not in potential_anomaly:
    #             anomaly_neighboor_detect = np.array(
    #                 cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 1, limit_size),
    #                 dtype=np.int32)
    #         else:
    #             anomaly_neighboor_detect = np.array(
    #                 cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 2, limit_size),
    #                 dtype=np.int32)
    #         if len(set(anomaly_neighboor_detect[:, 1]).intersection(potential_anomaly)) == 0:
    #             anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
    #                                          dtype=np.int32)
    #             potential_anomaly.extend([x[1] for x in anomaly_neighboor])
    #             for NN_pair in anomaly_neighboor:
    #                 Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
    #                     result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
    #         else:
    #             consider_point = np.max(
    #                 [i for i in list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
    #                  if
    #                  i not in potential_anomaly])
    #             if (raw_dta.value.values[anomaly_point] - raw_dta.value.values[
    #                 consider_point] - median_sec_der - std_sec_der > 0):
    #                 anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
    #                                              dtype=np.int32)
    #                 potential_anomaly.extend((x[1] for x in anomaly_neighboor))
    #                 for NN_pair in anomaly_neighboor:
    #                     Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[
    #                         0] * alpha if \
    #                         result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
    #             else:
    #                 result_dta.anomaly_score[anomaly_point] = 0
    #     else:
    #         consider_point = max(set(np.arange(anomaly_point)).difference(set(potential_anomaly)))
    #         if (abs(raw_dta.value.values[anomaly_point] - raw_dta.value.values[
    #             consider_point]) - median_sec_der - std_sec_der >= 0):
    #             anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
    #                                          dtype=np.int32)
    #             potential_anomaly.extend([x[1] for x in anomaly_neighboor])
    #             for NN_pair in anomaly_neighboor:
    #                 Y[NN_pair[1]] = Y[NN_pair[1]] + result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha if \
    #                     result_dta['anomaly_score'][anomaly_point] - NN_pair[0] * alpha > 0 else Y[NN_pair[1]]
    #         else:
    #             result_dta.anomaly_score[anomaly_point] = 0
    return sssss - time.time()


def online_anomaly_detection(result_dta, raw_dta, alpha, DATA_FILE):
    file_path_chart = "./active_result/all/" + DATA_FILE + "/" + DATA_FILE;
    median_sec_der = np.mean(result_dta['value'])
    std_sec_der = np.std(result_dta['value'])
    dict_neighbor = {}

    dta_full = result_dta

    dta_full.value.index = result_dta.timestamp

    std_anomaly_set = np.std(result_dta['anomaly_score'])
    np.argsort(result_dta['anomaly_score'])

    # ind = detect_peaks(result_dta['value'], show=True)
    # print(ind)
    # Get 5% anomaly point
    # anomaly_index =
    # np.array(np.argsort(result_dta['anomaly_score']))[-five_percentage:]
    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set])

    limit_size = int(1 / alpha)
    # Y is the anomaly spreding and Z is the normal spreading.
    Y = np.zeros(len(result_dta['anomaly_score']))
    Z = np.zeros(len(result_dta['anomaly_score']))
    X = list(map(lambda x: [x, result_dta.values[x][4]], np.arange(len(result_dta.values))))
    SAX_data = ts_to_string(znorm(np.array(result_dta['value'])), cuts_for_asize(10))

    tree = nb.KDTree(X, leaf_size=200, metric='euclidean')
    potential_anomaly = []
    executor_y = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )
    cmfunc.plot_data_all(file_path_chart,
                         [[list(range(0, len(raw_dta.value))), raw_dta.value],
                          [list([index for index, value in enumerate(raw_dta.anomaly_point.values) if value == 1]),
                           raw_dta.value[list(
                               [index for index, value in enumerate(raw_dta.anomaly_point.values) if value == 1])]],
                          [list([index for index, value in enumerate(raw_dta.change_point.values) if value == 1]),
                           raw_dta.value[list(
                               [index for index, value in enumerate(raw_dta.change_point.values) if value == 1])]],
                          [list([index for index, value in enumerate(raw_dta.anomaly_pattern.values) if value == 1]),
                           raw_dta.value[list(
                               [index for index, value in enumerate(raw_dta.anomaly_pattern.values) if value == 1])]]],
                         ['lines', 'markers', 'markers', 'markers'],
                         [None, 'x', 'circle', 'x'],
                         ['Raw data', "Detected Anomaly Point", "Detected Change Point",
                          "Detected Anomaly Patterm"]
                         )

    ################################### CHECKING THE POINT AGAIN ####################################
    # potential_anomaly = anomaly_index
    flag_running = True
    if flag_running == True:
        for anomaly_point in anomaly_index:
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

                else:
                    consider_point = np.max(
                        [i for i in
                         list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
                         if
                         i not in potential_anomaly])
                    if (raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                        consider_point] - median_sec_der - std_sec_der > 0):
                        anomaly_neighboor = np.array(
                            cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                            dtype=np.int32)
                        potential_anomaly.extend((x[1] for x in anomaly_neighboor))
                    else:
                        result_dta.anomaly_score[anomaly_point] = 0
            else:
                consider_point = max(set(np.arange(anomaly_point)).difference(set(potential_anomaly)))
                if (abs(raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                    consider_point]) - 3 * std_sec_der >= 0):
                    anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size),
                                                 dtype=np.int32)
                    potential_anomaly.extend([x[1] for x in anomaly_neighboor])
                else:
                    result_dta.anomaly_score[anomaly_point] = 0

    anomaly_index = [i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set]

    for i in anomaly_index:
        if find_element_in_list(i - 1, list(anomaly_index)) != None:
            anomaly_index.remove(i)
            result_dta.anomaly_score[i] = 0

    #anomaly_index = [i for i in anomaly_index if find_element_in_list(i - 1, list(anomaly_index)) == None]

    start_time_calculate_Y = time.time()

    ############################## CALCULATE SCORE ############################
    magnitude_score_array = []
    limit_size_conf = np.int(len(X) * 0.03)
    for i in anomaly_index:
        temp_score = []
        temp_inverse_neighbors = cmfunc.find_inverneghboor_of_point_2(tree, X, i, limit_size_conf)
        dict_neighbor[i] = temp_inverse_neighbors
        # temp_score.append(temp_inverse_neighbors > limit_size_conf)
        SAX_spreading_pattern = ''.join([SAX_data[i] for i in temp_inverse_neighbors[:, 1]])
        deveriate_SAX_spreading_pattern = ord(SAX_spreading_pattern[0]) - ord(SAX_spreading_pattern[-1])
        # sax_inverse_neighbors = ts_to_string(znorm(np.array(result_dta['value'][temp_inverse_neighbors[:,1]])), cuts_for_asize(20))
        magnitude_score_array.append(
            [i, deveriate_SAX_spreading_pattern, sorted(temp_inverse_neighbors[:, 1]), SAX_spreading_pattern])

    ################## Add the points have the same INN into a group ###########################
    final_magnitude_score_array = []
    for i in magnitude_score_array:
        temp_point_array = [i[0]]
        # for j in magnitude_score_array:
        #     if (i[2] == j[2]) and i[0] != j[0]:
        #         temp_point_array.append(j[0])
        if final_magnitude_score_array != []:
            if (sorted(temp_point_array) not in list(np.array(final_magnitude_score_array)[:, 0])):
                final_magnitude_score_array.append([sorted(temp_point_array), i[1], i[2], i[3]])
        else:
            final_magnitude_score_array.append([sorted(temp_point_array), i[1], i[2], i[3]])

    std_example_data = []
    std_example_outer = []
    detect_final_result = [[], []]
    for score_pattern in final_magnitude_score_array:
        detect_pattern = score_pattern[2]
        while checking_pattern_exist_max(score_pattern, detect_pattern, final_magnitude_score_array)[0] == True:
            finding_index = checking_pattern_exist_max(score_pattern, detect_pattern, final_magnitude_score_array)[1]
            common_point = list(set(detect_pattern).intersection(set(final_magnitude_score_array[finding_index][0])))
            for temp_common_point in common_point:
                detect_pattern.remove(temp_common_point)
            # detect_pattern.remove(max(detect_pattern))

        while checking_pattern_exist_min(score_pattern, detect_pattern, final_magnitude_score_array)[0] == True:
            finding_index = checking_pattern_exist_min(score_pattern, detect_pattern, final_magnitude_score_array)[1]
            common_point = list(set(detect_pattern).intersection(set(final_magnitude_score_array[finding_index][0])))
            for temp_common_point in common_point:
                detect_pattern.remove(temp_common_point)
            # detect_pattern.remove(min(detect_pattern))

        # while checking_pattern_exist(score_pattern,detect_pattern, final_magnitude_score_array)[0] == True:
        #     finding_index = checking_pattern_exist(score_pattern,detect_pattern, final_magnitude_score_array)[1]
        #     common_point = list(set(detect_pattern).intersection(set(final_magnitude_score_array[finding_index][2])))
        #     for temp_common_point  in common_point:
        #         detect_pattern.remove(temp_common_point)
        # detect_pattern.remove(min(detect_pattern))

        ##### max(detect_patter) must be not in final_magnitude_score_array checking #######
        example_data = [i for i in (
                list(raw_dta.value.values[
                         list(z for z in range(int(min(detect_pattern) - 3), int(min(detect_pattern))))]) + list(
            raw_dta.value.values[
                list(z for z in range(int(max(detect_pattern) + 1), int(max(detect_pattern) + 4)) if
                     z < len(raw_dta.value.values))]) + list(raw_dta.value.values[list(
            z for z in range(int(min(detect_pattern)), int(max(detect_pattern) + 1)) if z not in detect_pattern)]))]

        in_std_with_Anomaly = np.std(
            example_data + list(raw_dta.value.values[list(z for z in detect_pattern)]))
        std_example_data.append(in_std_with_Anomaly)

        example_data_iner = list(raw_dta.value.values[list(z for z in detect_pattern)])

        in_std_with_NonAnomaly = np.std(example_data)
        score_pattern.append(in_std_with_Anomaly / in_std_with_NonAnomaly)
        score_pattern.append(len(detect_pattern) / np.int(len(result_dta) * 0.02))

        SAX_spreading_pattern = ''.join([SAX_data[i] for i in detect_pattern])
        #deveriate_SAX_spreading_pattern = ord(SAX_spreading_pattern[0]) - ord(SAX_spreading_pattern[-1])
        deveriate_SAX_spreading_pattern = sum(cmfunc.change_after_k_seconds_with_abs([ord(i) for i in SAX_spreading_pattern], k=1))
        score_pattern[1] = deveriate_SAX_spreading_pattern

    ##############################------PLOTING-----##################################################

    from plotly import tools
    import plotly.plotly as py
    import plotly.graph_objs as go
    import plotly
    import pandas as pd

    plotly.tools.set_credentials_file(username='kimhung1990', api_key='Of6D1v3klVr2tWI2piK8')

    # scatter = dict(
    #     mode="markers",
    #     name="y",
    #     type="scatter3d",
    #     x=np.array(final_magnitude_score_array)[:, 1], y=np.array(final_magnitude_score_array)[:, 4],
    #     z=np.array(final_magnitude_score_array)[:, 5],
    #     marker=dict(size=2, color="rgb(23, 190, 207)")
    # )
    # clusters = dict(
    #     alphahull=7,
    #     name="y",
    #     opacity=0.1,
    #     type="mesh3d",
    #     x=np.array(final_magnitude_score_array)[:, 1], y=np.array(final_magnitude_score_array)[:, 4],
    #     z=np.array(final_magnitude_score_array)[:, 5]
    # )
    # layout = dict(
    #     title='3d point clustering',
    #     scene=dict(
    #         xaxis=dict(zeroline=False),
    #         yaxis=dict(zeroline=False),
    #         zaxis=dict(zeroline=False),
    #     )
    # )
    # fig = dict(data=[scatter], layout=layout)
    # Use py.iplot() for IPython notebook
    # plotly.offline.plot(fig, filename='3d point clustering')

    for i in final_magnitude_score_array:
        temp_detect_pattern_length = len(i[2])
        temp_detect_pattern_length_round = round(temp_detect_pattern_length,
                                                 -(len(str(
                                                     temp_detect_pattern_length)) - 1)) if temp_detect_pattern_length <= np.int(
            len(result_dta) * 0.02) else round(np.int(len(result_dta) * 0.03), -(len(str(np.int(len(result_dta) * 0.03))) - 1))

        if np.float(i[1])> 0:
            i[1] = 1
        if np.float(i[1]) == 0:
            i[1] = 0
        if np.float(i[1]) < 0:
            i[1] = -1

        i[1] = str(i[1]) + str(temp_detect_pattern_length_round)

    frequency_pattern_all = list(np.array(final_magnitude_score_array)[:, 1])

    for i in final_magnitude_score_array:
        frequency_pattern = frequency_pattern_all.count(i[1]) / len(final_magnitude_score_array)
        i.append(frequency_pattern)

    # max_anomaly_size = 1
    median_frequency = np.median(np.array(final_magnitude_score_array)[:, 6])

    flag_active_learning = 1
    magnitude_threshold = 1
    correlation_threshold = median_frequency
    varriance_threshold = 1.5
    dict_score = {}
    dict_labeled = {"anomaly": [], "anomaly_pattern": [], "change_point": []}

    for i in final_magnitude_score_array:
        temp_score_array = [i[4], i[5], i[6]]
        dict_score[i[0][0]] = temp_score_array

    backup_final_magnitude_score_array = list(final_magnitude_score_array)
    round_index_value = 0
    while flag_active_learning != 0:
        list_anomaly_pattern, list_anomaly_points, list_changed_point, list_uncertainty_point = active_learning_check(
            final_magnitude_score_array, correlation_threshold, magnitude_threshold, varriance_threshold)
        for z in dict_labeled['anomaly']:
            list_anomaly_points.append(z)
        for z in dict_labeled['anomaly_pattern']:
            list_anomaly_pattern.append(z)
        for z in dict_labeled['change_point']:
            list_changed_point.append(z)

        dict_result_analytic = {}
        for i in backup_final_magnitude_score_array:
            dict_result_analytic[i[0][0]] = i

        ground_anomaly_list = [index for index, value in enumerate(raw_dta['anomaly_pattern'].values) if value == 1]
        ground_anomaly_list.extend([index for index, value in enumerate(raw_dta['anomaly_point'].values) if value == 1])
        ground_change_point_list = [index for index, value in enumerate(raw_dta['change_point'].values) if value == 1]

        result_anomaly_list = []
        result_changepoint_list = []
        for j in list_anomaly_points:
            result_anomaly_list.append(j[0])
        for j in list_anomaly_pattern:
            result_anomaly_list.extend(dict_result_analytic[j[0]][2])
        for j in list_changed_point:
            result_changepoint_list.append(j[0])

        outF = open(file_path_chart + ".txt", "a")
        outF.write("Round " + str(round_index_value))
        outF.write("\n")
        outF.write("Anomaly Detection Correctness: " + str(
            len(set(ground_anomaly_list).intersection(set(result_anomaly_list))) / len(ground_anomaly_list)))
        outF.write("\n")
        outF.write("Change Point Detection Correctness: " + str(
            len(set(ground_change_point_list).intersection(set(result_changepoint_list))) / len(
                ground_change_point_list)))
        outF.write("\n")
        outF.write("--------------------------------------------")
        outF.write("\n")
        outF.close()
        round_index_value = round_index_value + 1

        print("Anomaly Detection Correctness: {}".format(
            len(set(ground_anomaly_list).intersection(set(result_anomaly_list))) / len(ground_anomaly_list)))
        print("Change Point Detection Correctness: {}".format(
            len(set(ground_change_point_list).intersection(set(result_changepoint_list))) / len(
                ground_change_point_list)))

        print("--------------------------------------------")

        if len(list_uncertainty_point) != 0:
            ####### Calculate the uncertain score for the list_uncertainty_point #############
            uncertainty_score_array = {}
            dict_uncertaity = {}
            asking_point = 0
            for uncertainty_point in list_uncertainty_point:
                temp_score_1 = dict_score[uncertainty_point[0]]
                ###### Calculate the different of this score to each kind of examing type ###########
                diff_with_anomalies = [
                    0 if temp_score_1[0] > varriance_threshold else (varriance_threshold - temp_score_1[
                        0]) / varriance_threshold,
                    0 if temp_score_1[1] == 1 / np.int(len(result_dta) * 0.02) else np.abs(
                        (magnitude_threshold - temp_score_1[1])) / magnitude_threshold,
                    0 if temp_score_1[2] < correlation_threshold else (temp_score_1[
                                                                           2] - correlation_threshold) / correlation_threshold
                ]
                diff_with_pattern = [
                    0 if temp_score_1[0] > varriance_threshold else (varriance_threshold - temp_score_1[
                        0]) / varriance_threshold,
                    0 if temp_score_1[1] < magnitude_threshold else (temp_score_1[
                                                                         1] - magnitude_threshold) / magnitude_threshold,
                    0 if temp_score_1[2] < correlation_threshold else (temp_score_1[
                                                                           2] - correlation_threshold) / correlation_threshold
                ]
                diff_with_changepoint = [
                    0 if temp_score_1[0] <= varriance_threshold else (temp_score_1[
                                                                          0] - varriance_threshold) / varriance_threshold,
                    0 if temp_score_1[1] >= magnitude_threshold else (magnitude_threshold - temp_score_1[
                        1]) / magnitude_threshold,
                    0 if temp_score_1[2] >= correlation_threshold else (correlation_threshold - temp_score_1[
                        2]) / correlation_threshold
                ]
                dict_uncertaity[uncertainty_point[0]] = np.min(
                    [np.sum(diff_with_anomalies), np.sum(diff_with_pattern), np.sum(diff_with_changepoint)])
            asking_point = np.argmax([dict_uncertaity[i] for i in dict_uncertaity])

            cmfunc.plot_data_all(file_path_chart + "_" + str(round_index_value),
                                 [[list(range(0, len(raw_dta.value))), raw_dta.value],
                                  [result_anomaly_list,
                                   raw_dta.value[list(result_anomaly_list)]],
                                  [list(result_changepoint_list),
                                   raw_dta.value[list(result_changepoint_list)]],
                                  [list(np.array(list_uncertainty_point).flatten()),
                                   raw_dta.value[list(np.array(list_uncertainty_point).flatten())]]
                                  ],
                                 ['lines', 'markers', 'markers', 'markers'],
                                 [None, 'x', 'circle', 'circle'],
                                 ['Raw data', "Detected Anomaly Point", "Detected Change Point", "Uncertain Point"]
                                 )

            point_type = input("Please enter the type of point " + str(list_uncertainty_point[asking_point]))
            print("You entered " + str(point_type))
            if str(point_type) == '1':
                dict_labeled['anomaly'].append(list_uncertainty_point[asking_point])
                examing_point_index = list(np.array(final_magnitude_score_array)[:, 0]).index(
                    list_uncertainty_point[asking_point])
                if final_magnitude_score_array[examing_point_index][4] < varriance_threshold:
                    varriance_threshold = final_magnitude_score_array[examing_point_index][4]
            if str(point_type) == '2':
                dict_labeled['anomaly_pattern'].append(list_uncertainty_point[asking_point])
                examing_point_index = list(np.array(final_magnitude_score_array)[:, 0]).index(
                    list_uncertainty_point[asking_point])
                if final_magnitude_score_array[examing_point_index][4] < varriance_threshold:
                    varriance_threshold = final_magnitude_score_array[examing_point_index][4]
                if final_magnitude_score_array[examing_point_index][5] > magnitude_threshold:
                    magnitude_threshold = final_magnitude_score_array[examing_point_index][5]
                if final_magnitude_score_array[examing_point_index][6] > correlation_threshold:
                    correlation_threshold = final_magnitude_score_array[examing_point_index][6]
            if str(point_type) == '3':
                dict_labeled['change_point'].append(list_uncertainty_point[asking_point])
                examing_point_index = list(np.array(final_magnitude_score_array)[:, 0]).index(
                    list_uncertainty_point[asking_point])
                if final_magnitude_score_array[examing_point_index][4] >= varriance_threshold:
                    varriance_threshold = final_magnitude_score_array[examing_point_index][4]
                if final_magnitude_score_array[examing_point_index][5] <= magnitude_threshold:
                    magnitude_threshold = final_magnitude_score_array[examing_point_index][5]
                if final_magnitude_score_array[examing_point_index][6] <= correlation_threshold:
                    correlation_threshold = final_magnitude_score_array[examing_point_index][6]
            ####### Remove the labeled point in final_magnitude_score_array #########
            del final_magnitude_score_array[examing_point_index]


        else:
            flag_active_learning = 0
            ########## Plot Result ################
            cmfunc.plot_data_all(DATA_FILE,
                                 [[list(range(0, len(raw_dta.value))), raw_dta.value],
                                  [result_anomaly_list,
                                   raw_dta.value[list(result_anomaly_list)]],
                                  [list(result_changepoint_list),
                                   raw_dta.value[list(result_changepoint_list)]]
                                  ],
                                 ['lines', 'markers', 'markers'],
                                 [None, 'x', 'circle'],
                                 ['Raw data', "Detected Anomaly Point", "Detected Change Point"]
                                 )

    # list_anomaly_points = [i[0] for i in final_magnitude_score_array if i[4]>2 and i[5] == len(i[0]) and i[1] == 0]
    # list_anomaly_pattern = [i[0] for i in final_magnitude_score_array if i[4]>2 and i[5] < max_anomaly_size and i[6] <= median_frequency]
    # list_changed_point = [i[0] for i in final_magnitude_score_array if i[4]<1.5 and i[5] >= max_anomaly_size]
    # list_uncertainty_point = [i[0] for i in final_magnitude_score_array if i[0] not in list_anomaly_points and i[0] not in list_changed_point and i[0] not in list_anomaly_pattern]

    #print([[i[0], i[1], i[4], i[5], i[6]] for i in final_magnitude_score_array])


    ######################################################
    # Calculate Y
    # Asyncronous function to calculate Y value
    for detected_change_point in result_changepoint_list:
        result_dta.anomaly_score[detected_change_point] = 0
    async def calculate_Y_value_big(executor):
        loop = asyncio.get_event_loop()
        blocking_tasks = [
            loop.run_in_executor(executor, calculate_Y_value, alpha, anomaly_point, limit_size, median_sec_der,
                                 dict_neighbor, raw_dta, result_dta, std_sec_der, tree, X, Y)
            for anomaly_point in sorted(result_anomaly_list)
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

    # ssss = time.time()
    # # normal_index = [i for i, value in enumerate(result_dta['anomaly_score']) if
    # #                filter_function_z(result_dta['anomaly_score'],i,limit_size/2,data_length) > 0 and value <= np.percentile(result_dta['anomaly_score'], 20)]
    # anomaly_score = result_dta['anomaly_score']
    # normal_index = [i for i, value in enumerate(anomaly_score) if
    #                 value <= 0 and anomaly_score[tree.query([X[i]], k=int(limit_size / 2))[1][0]].values.max() > 0]
    # print("Chossing time for {}: {}".format(len(normal_index), time.time() - ssss))
    # normal_points_array = chunks(sorted(normal_index), 30)
    #
    # async def calculate_z_value(executor):
    #     loop = asyncio.get_event_loop()
    #     blocking_tasks = [
    #         loop.run_in_executor(executor, find_inverneghboor_of_point_blocking, alpha, normal_points,
    #                              result_dta, Z)
    #         for normal_points in normal_points_array
    #     ]
    #     completed, pending = await asyncio.wait(blocking_tasks)
    #     results = [t.result() for t in completed]
    #     print("The sum times of calculating Z: {}".format(sum(results)))
    #
    # event_loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(event_loop)
    # try:
    #     event_loop.run_until_complete(
    #         calculate_z_value(executor)
    #     )
    # finally:
    #     event_loop.close()
    #
    # result_dta.anomaly_score = result_dta.anomaly_score - Z
    #
    # end_time_calculate_Z = time.time()
    # print("Calculating Z Time: {}".format(time.time() - ssss))

    final_score = list(map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score))
    final_score = (final_score - np.min(final_score)) / (np.max(final_score) - np.min(final_score))

    # Calculating Change point.
    start_time_calculate_changepoint = time.time()

    ### Find potential anomaly point
    std_final_point = np.std(final_score)

    anomaly_set = [i for i, v in enumerate(final_score) if v > 0]

    # The algorithm to seperate anomaly point and change point.
    file_path = "./active_result/all/" + DATA_FILE + "/" + DATA_FILE + ".csv"
    df = pd.read_csv(file_path)
    df = df.assign(anomaly_score=pd.Series(final_score).values)
    del df['anomaly_pattern']
    del df['anomaly_point']
    del df['change_point']

    ts = time.time()
    time_array = [datetime.datetime.fromtimestamp(ts - 10000 * i).strftime('%Y-%m-%d %H:%M:%S') for i in
                  np.arange(0, len(df['value']))]
    temp_time_stamp = pd.to_datetime(time_array, format='%Y-%m-%d %H:%M:%S').sort_values()
    df['timestamp'] = pd.Series(temp_time_stamp).values
    df = df.assign(label=pd.Series(np.zeros(len(df))).values)
    df.sort_values(by='timestamp', inplace=True)
    df.sort_index(axis=0, inplace=True)

    df.to_csv(file_path, index=False);
    df.to_csv(os.path.normpath('D:/Google Drive/13. These cifre/Data Cleaning/workspace/NAB/results/myAL/realKnownCause/myAL_' + DATA_FILE.replace(" ", "_") + "_sorted.csv"), index=False)

    return detect_final_result
    # return [detect_final_result,chartmess]


def checking_pattern_exist_max(score_pattern, detect_pattern, final_magnitude_score_array):
    flag_checking_pattern = False
    possition_index = -1
    if len(detect_pattern) != 0:
        for temp_pattern_index, temp_pattern in enumerate(final_magnitude_score_array):
            if temp_pattern != score_pattern and find_element_in_list(max(detect_pattern),
                                                                      list(temp_pattern[0])) != None:
                possition_index = temp_pattern_index
                flag_checking_pattern = True
                break
    return [flag_checking_pattern, possition_index]


def checking_pattern_exist_min(score_pattern, detect_pattern, final_magnitude_score_array):
    flag_checking_pattern = False
    possition_index = -1
    if len(detect_pattern) != 0:
        for temp_pattern_index, temp_pattern in enumerate(final_magnitude_score_array):
            if temp_pattern != score_pattern and find_element_in_list(min(detect_pattern),
                                                                      list(temp_pattern[0])) != None:
                possition_index = temp_pattern_index
                flag_checking_pattern = True
                break
    return [flag_checking_pattern, possition_index]


def checking_pattern_exist(score_pattern, detect_pattern, final_magnitude_score_array):
    flag_checking_pattern = False
    possition_index = -1
    for temp_pattern_index, temp_pattern in enumerate(final_magnitude_score_array):
        if temp_pattern != score_pattern and set(temp_pattern[2]).issubset(set(detect_pattern)) == True:
            possition_index = temp_pattern_index
            flag_checking_pattern = True
            break
    return [flag_checking_pattern, possition_index]


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


def active_learning_check(final_magnitude_score_array, correlation_threshold, magnitude_threshold,
                          varriance_threshold):
    list_anomaly_points = []
    list_anomaly_pattern = []
    list_changed_point = []
    list_uncertainty_point = []
    for i in final_magnitude_score_array:
        if i[4] >= varriance_threshold and i[1] == '01':
            list_anomaly_points.append(i[0])
        if i[4] >= varriance_threshold and i[5] <= magnitude_threshold and i[6] <= correlation_threshold:
            list_anomaly_pattern.append(i[0])
        if i[4] <= varriance_threshold and i[5] >= magnitude_threshold and i[6] >= correlation_threshold:
            list_changed_point.append(i[0])
        if i[0] not in list_anomaly_points and i[0] not in list_changed_point and i[0] not in list_anomaly_pattern:
            list_uncertainty_point.append(i[0])
    return list_anomaly_pattern, list_anomaly_points, list_changed_point, list_uncertainty_point
