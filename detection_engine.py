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

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from saxpy.saxpy.znorm import znorm
from saxpy.saxpy.sax import ts_to_string
from saxpy.saxpy.alphabet import cuts_for_asize

from scripts.detect_peaks import detect_peaks

Maggitute_radion = 0.02

varian_radion = 2

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


def online_anomaly_detection(result_dta, raw_dta, alpha, DATA_FILE, confident_threshold):
    file_path_chart = "./active_result/all/" + DATA_FILE + "/" + DATA_FILE;
    file_path = "./active_result/all/" + DATA_FILE + "/" + DATA_FILE + ".csv"
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
    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > std_anomaly_set])
    #anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 0])

    #limit_size = int(1 / alpha)

    # Y is the anomaly spreding and Z is the normal spreading.
    Y = np.zeros(len(result_dta['anomaly_score']))
    Z = np.zeros(len(result_dta['anomaly_score']))
    #X = list(map(lambda x: [result_dta.values[x][4]], np.arange(len(result_dta.values))))
    X = list(map(lambda x: [10*x, result_dta.values[x][4]], np.arange(len(result_dta.values))))
    SAX_data = ts_to_string(znorm(np.array(result_dta['value'])), cuts_for_asize(10))
    limit_size_conf = np.int(len(X) * Maggitute_radion)
    tree = nb.KDTree(X, leaf_size=200, metric='euclidean')

    potential_anomaly = []
    executor_y = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
    )

    ################################### CHECKING THE POINT AGAIN ####################################
    # potential_anomaly = anomaly_index
    flag_running = True
    if flag_running == True:
        for anomaly_point in anomaly_index:
            if anomaly_point - 1 not in potential_anomaly and anomaly_point - 2 not in potential_anomaly:
                if anomaly_point - 1 not in potential_anomaly:
                    anomaly_neighboor_detect = np.array(
                        cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 1, limit_size_conf),
                        dtype=np.int32)
                else:
                    anomaly_neighboor_detect = np.array(
                        cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point - 2, limit_size_conf),
                        dtype=np.int32)
                if len(anomaly_neighboor_detect[:, 1]) != 0:
                    if len(set(anomaly_neighboor_detect[:, 1]).intersection(potential_anomaly)) == 0:
                        anomaly_neighboor = np.array(
                            cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size_conf),
                            dtype=np.int32)
                        if len(anomaly_neighboor[0]) != 0:
                            potential_anomaly.extend([x[1] for x in anomaly_neighboor])

                    else:
                        if (len([i for i in
                             list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
                             if
                             i not in potential_anomaly]) != 0):
                            consider_point = np.max([i for i in
                                 list(set(range(0, anomaly_point - 1)).difference(set(anomaly_neighboor_detect[:, 1])))
                                 if
                                 i not in potential_anomaly]
                                )
                            if (raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                                consider_point] - median_sec_der - std_sec_der > 0):
                                anomaly_neighboor = np.array(
                                    cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size_conf),
                                    dtype=np.int32)
                                potential_anomaly.extend((x[1] for x in anomaly_neighboor))
                            else:
                                result_dta.anomaly_score[anomaly_point] = 0

            else:
                if (len(set(np.arange(anomaly_point)).difference(set(potential_anomaly))) != 0):
                    consider_point = max(set(np.arange(anomaly_point)).difference(set(potential_anomaly)))
                    if (abs(raw_dta.value.values[anomaly_point] - raw_dta.value.values[
                        consider_point]) - 3 * std_sec_der >= 0):
                        anomaly_neighboor = np.array(cmfunc.find_inverneghboor_of_point(tree, X, anomaly_point, limit_size_conf),
                                                     dtype=np.int32)
                        if len(anomaly_neighboor[0]) !=0:
                            potential_anomaly.extend([x[1] for x in anomaly_neighboor])
                    else:
                        result_dta.anomaly_score[anomaly_point] = 0

    #anomaly_index = [i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set]

    # for i in anomaly_index:
    #     if find_element_in_list(i - 1, list(anomaly_index)) != None:
    #         anomaly_index.remove(i)
    #         result_dta.anomaly_score[i] = 0



    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > std_anomaly_set])
    anomaly_index = [i for i in anomaly_index if find_element_in_list(i - 1, list(anomaly_index)) == None]
    start_time_calculate_Y = time.time()

    ############################## CALCULATE SCORE ############################
    magnitude_score_array = []
    #limit_size_conf = np.int(len(X) * 0.03)
    for i in anomaly_index:
        temp_score = []
        temp_inverse_neighbors = cmfunc.find_inverneghboor_of_point_2(tree, X, i, limit_size_conf)
        if len(temp_inverse_neighbors[0]) != 0:
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

        ################## Calculate Variance Score #########################
        in_std_with_NonAnomaly = np.std(example_data)
        if in_std_with_Anomaly >= in_std_with_NonAnomaly * varian_radion:
            score_pattern.append(1)
        else:
            score_pattern.append(
                float(in_std_with_Anomaly / in_std_with_NonAnomaly / varian_radion) if in_std_with_NonAnomaly != 0 else 999)
        ################## Calculate Magitutde Score #########################
        if len(detect_pattern) >= np.int(len(result_dta) * Maggitute_radion):
            score_pattern.append(1)
        else:
            if len(detect_pattern) == 1:
                score_pattern.append(0)
            else:
                score_pattern.append(len(detect_pattern) / np.int(len(result_dta) * Maggitute_radion))

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

    for i in final_magnitude_score_array:
        temp_detect_pattern_length = len(i[2])
        temp_detect_pattern_length_round = round(temp_detect_pattern_length,
                                                 -(len(str(
                                                     temp_detect_pattern_length)) - 1)) if temp_detect_pattern_length <= np.int(
            len(result_dta) * Maggitute_radion) else round(np.int(len(result_dta) * Maggitute_radion), -(len(str(np.int(len(result_dta) * Maggitute_radion))) - 1))

        if np.float(i[1])> 0:
            i[1] = 1
        if np.float(i[1]) == 0:
            i[1] = 0
        if np.float(i[1]) < 0:
            i[1] = -1

        i[1] = str(i[1]) + str(temp_detect_pattern_length_round)

    frequency_pattern_all = list(np.array(final_magnitude_score_array)[:, 1])

    ################## Calculate Correlation Score #########################

    for i in final_magnitude_score_array:
        frequency_pattern = frequency_pattern_all.count(i[1]) / len(final_magnitude_score_array)
        i.append(frequency_pattern)

    # max_anomaly_size = 1
    median_frequency = np.median(np.array(final_magnitude_score_array)[:, 6])

    flag_active_learning = 1
    magnitude_threshold = 1
    correlation_threshold = median_frequency
    varriance_threshold = 1.5
    array_evaluation_result = {'precision': [],
        'min_confident': [],'total_points':[]
    }
    array_evaluation_result_myAL = {'recall_anomaly': [],
                               'precision_anomaly': [],
                               'recall_changePoint': [],
                               'precision_changePoint': []}

    ground_anomaly_list = [index for index, value in enumerate(raw_dta['anomaly_pattern'].values) if value == 1]
    ground_anomaly_list.extend([index for index, value in enumerate(raw_dta['anomaly_point'].values) if value == 1])
    ground_change_point_list = [index for index, value in enumerate(raw_dta['change_point'].values) if value == 1]

    dict_score = {}
    dict_labeled = {"anomaly": [], "anomaly_pattern": [], "change_point": []}


    for i in final_magnitude_score_array:
        #temp_score_array = [i[4], i[5], i[6]]
        temp_score_array = [i[4], i[5]]
        dict_score[i[0][0]] = temp_score_array

    backup_final_magnitude_score_array = list(final_magnitude_score_array)
    round_index_value = 0

    X_data = np.array(list(dict_score.values()))
    Y_data = np.array(result_dta['change_point'][[i for i in dict_score]] * 3 + result_dta['anomaly_pattern'][
        [i for i in dict_score]] * 2 + result_dta['anomaly_point'][[i for i in dict_score]] * 1)

    X_pool = np.array(list(dict_score.values()))
    Y_pool = np.array(result_dta['change_point'][[i for i in dict_score]]*3 + result_dta['anomaly_pattern'][[i for i in dict_score]]*2 + result_dta['anomaly_point'][[i for i in dict_score]]*1)

    #X_train = [[0.2, 1],[1, 0.1],[1, 0],[0,1]]
    X_train = [[0.2, 1],[1, 0.1],[1, 0]]
    #X_train = [[0.2, 1, 0.8],[1, 0.1, 0.1],[1, 0, 0.1]]
    Y_train = [3,2,1]


    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=uncertainty_sampling,
        X_training=X_train, y_training=Y_train
    )


    print('_______________________________________________')
    print('Accuracy before active learning: %f' % learner.score(X_data, Y_data))
    array_evaluation_result['precision'].append(learner.score(X_data, Y_data))
    array_evaluation_result['min_confident'].append(min([max(i) for i in learner.predict_proba(X_pool)]))
    array_evaluation_result['total_points'].append(len(X_data))

    dict_result_analytic = {}
    for i in backup_final_magnitude_score_array:
        dict_result_analytic[i[0][0]] = i

    n_queries = int(len(Y_data)/3)

    flag_asking = 1
    idx = 0

    before_activelearning_result = display_performance(X_data, array_evaluation_result_myAL, dict_result_analytic, dict_score, ground_anomaly_list,
                        ground_change_point_list, learner)

    while flag_asking == 1:
        query_idx, query_instance = learner.query(X_pool)
        learner.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=Y_pool[query_idx].reshape(1, )
        )
        idx = idx + 1
        # remove queried instance from pool
        #print("----------------Round {}----------------------------".format(idx))
        X_pool = np.delete(X_pool, query_idx, axis=0)
        Y_pool = np.delete(Y_pool, query_idx)
        if len(X_pool) != 0:
            #print('Accuracy after %d query : %f' % (idx, learner.score(X_data, Y_data)))
            proba_score = learner.predict_proba(X_pool);
            if min([max(i) for i in proba_score]) >= confident_threshold:
                flag_asking = 0
            #print('Min confident after %d query : %f' % (idx, min([max(i) for i in proba_score])))

            array_evaluation_result['precision'].append(learner.score(X_data, Y_data))
            array_evaluation_result['min_confident'].append(min([max(i) for i in proba_score]))
            array_evaluation_result['total_points'].append(len(X_data))

            ### Calculate the correcness
            # display_performance(X_data, array_evaluation_result_myAL, dict_result_analytic, dict_score,
            #                     ground_anomaly_list,
            #                     ground_change_point_list, learner)
        else:
            array_evaluation_result['precision'].append(learner.score(X_data, Y_data))
            array_evaluation_result['min_confident'].append(100)
            array_evaluation_result['total_points'].append(len(X_data))

            ### Calculate the correcness
            # display_performance(X_data, array_evaluation_result_myAL, dict_result_analytic, dict_score,
            #                     ground_anomaly_list,
            #                     ground_change_point_list, learner)
            flag_asking = 0


    # with plt.style.context('seaborn-white'):
    #     plt.figure(figsize=(7, 7))
    #     prediction = learner.predict(X_data)
    #     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    #     plt.title(
    #         'Classification accuracy after %i queries: %f' % (n_queries, learner.score(X_data, Y_data)))
    #     plt.show()

    # ###################################################### Write the result #########################################


    ### Calculate the correcness
    print('______________________Final Result_________________________')
    after_activerlerning_result = display_performance(X_data, array_evaluation_result_myAL, dict_result_analytic, dict_score, ground_anomaly_list,
                        ground_change_point_list, learner)

    ##########################################################################

    # df_final_evaluation = pd.read_csv(os.path.normpath(
    #     'D:/Google Drive/13. These cifre/Data Cleaning/workspace/result_yahoo_dataset_steps.csv'))
    #
    # df_final_evaluation.insert(0,"precision"+DATA_FILE,pd.Series(array_evaluation_result_myAL['precision_anomaly']), allow_duplicates=True)
    # df_final_evaluation.insert(0,"recall"+DATA_FILE,pd.Series(array_evaluation_result_myAL['recall_anomaly']), allow_duplicates=True)
    # df_final_evaluation.insert(0,"nb_anomalies"+DATA_FILE,pd.Series(np.full((1,len(array_evaluation_result_myAL['precision_anomaly'])),len(ground_anomaly_list), dtype=int)[0]), allow_duplicates=True)
    #
    # df_final_evaluation.to_csv(os.path.normpath(
    #     'D:/Google Drive/13. These cifre/Data Cleaning/workspace/result_yahoo_dataset_steps.csv'), index=False);

    #############################################################################################

    # df_final_result = pd.read_csv(os.path.normpath(
    #     'D:/Google Drive/13. These cifre/Data Cleaning/workspace/result_yahoo_dataset.csv'))
    # pd.DataFrame(data=d)
    if (after_activerlerning_result[0]+after_activerlerning_result[1]) == 0:
        end_result = [0,
                      2 * after_activerlerning_result[2] * after_activerlerning_result[3] / (
                              after_activerlerning_result[2] + after_activerlerning_result[3])]
    elif (after_activerlerning_result[2]+after_activerlerning_result[3]) == 0:
        end_result = [2 * after_activerlerning_result[0] * after_activerlerning_result[1] / (after_activerlerning_result[0]+after_activerlerning_result[1]),
                      0]
    else:
        end_result = [2 * after_activerlerning_result[0] * after_activerlerning_result[1] / (
                    after_activerlerning_result[0] + after_activerlerning_result[1]),
                      2 * after_activerlerning_result[2] * after_activerlerning_result[3] / (
                              after_activerlerning_result[2] + after_activerlerning_result[3])]

    # if end_result[0] + end_result[1] > max_AN + max_CP:
    #     max_AN = end_result[0]
    #     max_CP = end_result[1]
    #     result_dta.to_csv(file_path, index=False);
    #     df_percentage_result = pd.DataFrame(array_evaluation_result,
    #                                         columns=['precision', 'min_confident', 'total_points'])
    #     df_percentage_result.to_csv(file_path_chart + "_accuracy_model.csv")
    #     df_final_result = pd.DataFrame(data={'dataset': DATA_FILE,
    #                                          'bf_pre_anomaly': before_activelearning_result[0],
    #                                          'bf_re_anomaly': before_activelearning_result[1],
    #                                          'bf_pre_change': before_activelearning_result[2],
    #                                          'bf_re_change': before_activelearning_result[3],
    #                                          'af_pre_anomaly': after_activerlerning_result[0],
    #                                          'af_re_anomaly': after_activerlerning_result[1],
    #                                          'af_pre_change': after_activerlerning_result[2],
    #                                          'af_re_change': after_activerlerning_result[3],
    #                                          'nb_anomalies': len(ground_anomaly_list),
    #                                          'nb_change_point': len(ground_change_point_list)}, index=[0])
    #     df_final_result.to_csv(file_path_chart + "_result.csv");
    #
    #     cmfunc.plot_data_all(file_path_chart,
    #                          [[list(range(0, len(raw_dta.value))), raw_dta.value],
    #                           [list([index for index, value in enumerate(raw_dta.anomaly_point.values) if value == 1]),
    #                            raw_dta.value[list(
    #                                [index for index, value in enumerate(raw_dta.anomaly_point.values) if value == 1])]],
    #                           [list([index for index, value in enumerate(raw_dta.change_point.values) if value == 1]),
    #                            raw_dta.value[list(
    #                                [index for index, value in enumerate(raw_dta.change_point.values) if value == 1])]],
    #                           [list(
    #                               [index for index, value in enumerate(raw_dta.anomaly_pattern.values) if value == 1]),
    #                            raw_dta.value[list(
    #                                [index for index, value in enumerate(raw_dta.anomaly_pattern.values) if
    #                                 value == 1])]]],
    #                          ['lines', 'markers', 'markers', 'markers'],
    #                          [None, 'x', 'circle', 'x'],
    #                          ['Raw data', "Detected Anomaly Point", "Detected Change Point",
    #                           "Detected Anomaly Patterm"]
    #                          )


    # ######################################################
    # # Calculate Y
    # # Asyncronous function to calculate Y value
    # for detected_change_point in result_changepoint_list:
    #     result_dta.anomaly_score[detected_change_point] = 0
    # async def calculate_Y_value_big(executor):
    #     loop = asyncio.get_event_loop()
    #     blocking_tasks = [
    #         loop.run_in_executor(executor, calculate_Y_value, alpha, anomaly_point, limit_size, median_sec_der,
    #                              dict_neighbor, raw_dta, result_dta, std_sec_der, tree, X, Y)
    #         for anomaly_point in sorted(result_anomaly_list)
    #     ]
    #     completed, pending = await asyncio.wait(blocking_tasks)
    #     results = [t.result() for t in completed]
    #     print("The sum times of calculating Y: {}".format(sum(results)))
    #
    # # Starting calculating function
    # # loop = asyncio.new_event_loop()
    # # asyncio.set_event_loop(loop)
    # event_loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(event_loop)
    # try:
    #     event_loop.run_until_complete(
    #         calculate_Y_value_big(executor_y)
    #     )
    # finally:
    #     event_loop.close()
    #     print("Finish threading calculation of Y")
    #
    # # Calculate final score
    # result_dta.anomaly_score = result_dta.anomaly_score + Y
    #
    # print("Calculating Y Time for {} elements: {}".format(len(anomaly_index), time.time() - start_time_calculate_Y))
    #
    # #######################################################
    # ########## STARTING CALCULATING Z ####################
    #
    # # ssss = time.time()
    # # # normal_index = [i for i, value in enumerate(result_dta['anomaly_score']) if
    # # #                filter_function_z(result_dta['anomaly_score'],i,limit_size/2,data_length) > 0 and value <= np.percentile(result_dta['anomaly_score'], 20)]
    # # anomaly_score = result_dta['anomaly_score']
    # # normal_index = [i for i, value in enumerate(anomaly_score) if
    # #                 value <= 0 and anomaly_score[tree.query([X[i]], k=int(limit_size / 2))[1][0]].values.max() > 0]
    # # print("Chossing time for {}: {}".format(len(normal_index), time.time() - ssss))
    # # normal_points_array = chunks(sorted(normal_index), 30)
    # #
    # # async def calculate_z_value(executor):
    # #     loop = asyncio.get_event_loop()
    # #     blocking_tasks = [
    # #         loop.run_in_executor(executor, find_inverneghboor_of_point_blocking, alpha, normal_points,
    # #                              result_dta, Z)
    # #         for normal_points in normal_points_array
    # #     ]
    # #     completed, pending = await asyncio.wait(blocking_tasks)
    # #     results = [t.result() for t in completed]
    # #     print("The sum times of calculating Z: {}".format(sum(results)))
    # #
    # # event_loop = asyncio.new_event_loop()
    # # asyncio.set_event_loop(event_loop)
    # # try:
    # #     event_loop.run_until_complete(
    # #         calculate_z_value(executor)
    # #     )
    # # finally:
    # #     event_loop.close()
    # #
    # # result_dta.anomaly_score = result_dta.anomaly_score - Z
    # #
    # # end_time_calculate_Z = time.time()
    # # print("Calculating Z Time: {}".format(time.time() - ssss))
    #
    # final_score = list(map(lambda x: 0 if x < 0 else x, result_dta.anomaly_score))
    # final_score = (final_score - np.min(final_score)) / (np.max(final_score) - np.min(final_score))
    #
    # # Calculating Change point.
    # start_time_calculate_changepoint = time.time()
    #
    # ### Find potential anomaly point
    # std_final_point = np.std(final_score)
    #
    # # anomaly_set = [i for i, v in enumerate(final_score) if v > 0]
    # #
    # # # The algorithm to seperate anomaly point and change point.
    # # file_path = "./active_result/all/" + DATA_FILE + "/" + DATA_FILE + ".csv"
    # # df_null = pd.read_csv(os.path.normpath('D:/Google Drive/13. These cifre/Data Cleaning/workspace/NAB/results/null/realKnownCause/null_' + DATA_FILE.replace(" ", "_") + "_sorted.csv"))
    # #
    # # df = pd.read_csv(file_path)
    # # df = df.assign(anomaly_score=pd.Series(final_score).values)
    # # del df['anomaly_pattern']
    # # del df['anomaly_point']
    # # del df['change_point']
    # #
    # # df['timestamp'] = pd.Series(df_null['timestamp']).values
    # # df = df.assign(label=pd.Series(np.zeros(len(df))).values)
    # # df.sort_values(by='timestamp', inplace=True)
    # # df.sort_index(axis=0, inplace=True)
    # #
    # # #df.to_csv(file_path, index=False);
    # # df.to_csv(os.path.normpath('D:/Google Drive/13. These cifre/Data Cleaning/workspace/NAB/results/myAL/realKnownCause/myAL_' + DATA_FILE.replace(" ", "_") + "_sorted.csv"), index=False)

    return [end_result,idx, after_activerlerning_result[4]]
    # return [detect_final_result,chartmess]


def generate_tsing_data_format(DATA_FILE, df_raw, active_learning):


    file_path = "D:/Workspace/imr/data/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = df_raw['value'].values
    list_change_points = [i for i, x in enumerate(df_raw['change_point'].values) if x == 1]
    list_anonaly_points = [i for i, x in enumerate(df_raw['anomaly_point'].values) if x == 1]
    list_anomaly_pattern = [i for i, x in enumerate(df_raw['anomaly_pattern'].values) if x == 1]
    list_truth = df_raw['value'].values.copy()
    for i in list_anonaly_points:
        list_truth[i] = list_truth[i-1]
    for i in list_anomaly_pattern:
        list_truth[i] = list_truth[i-1]

    change_points_data = np.zeros(len(data))
    change_points_data[list_change_points[0:-1]] = 1

    anomaly_points_data = np.zeros(len(data))
    anomaly_points_data[list_anonaly_points] = 1
    anomaly_points_data[list_anomaly_pattern] = 1

#######################################################################################################################
    label_data = data.copy();
    doLabel = np.full((1, len(data)), False)[0]
    d = {'index': np.arange(1, len(data) + 1), 'value': data, 'label': label_data,
         'truth': list_truth, 'doLabel': doLabel, 'anomaly': anomaly_points_data}
    df = pd.DataFrame(data=d)
    df = df[['index', 'value', 'label', 'truth', 'doLabel', 'anomaly']]
    df.to_csv(file_path + "/" + DATA_FILE + "_original.csv", index=False);

############################################---LABEL BASED ON TRUST############################################################
    label_data_cabd = data.copy();
    doLabel_cabd = np.full((1, len(data)), False)[0]
    for index, value in enumerate(list_truth):
        if index in list_change_points:
            doLabel_cabd[index] = True
            doLabel_cabd[index - 1] = True
        if index in list_anomaly_pattern or index in list_anonaly_points:
            label_data_cabd[index] = value
            doLabel_cabd[index] = True
    d = {'index': np.arange(1, len(data) + 1), 'value': data, 'label': label_data_cabd,
         'truth': list_truth, 'doLabel': doLabel_cabd, 'anomaly': anomaly_points_data}
    df = pd.DataFrame(data=d)
    df = df[['index', 'value', 'label', 'truth', 'doLabel', 'anomaly']]
    df.to_csv(file_path + "/" + DATA_FILE + "_label_correct.csv", index=False)

############################################---LABEL BASED ON DETECTION______############################################################
    label_data_cabd_all = data.copy();
    doLabel_cabd_all = np.full((1, len(data)), False)[0]
    quality_require = 0.7
    for detect_result in active_learning:
        if detect_result[1] == 3 and detect_result[2] >=quality_require:
            doLabel_cabd_all[detect_result[0]] = True
            doLabel_cabd_all[detect_result[0] - 1] = True
        if detect_result[1] == 1 and detect_result[2] >= quality_require:
            label_data_cabd_all[detect_result[0]] = list_truth[detect_result[0]]
            doLabel_cabd_all[detect_result[0]] = True
        if detect_result[1] == 2 and detect_result[2] >= quality_require:
            label_data_cabd_all[detect_result[0]] = list_truth[detect_result[0]]
            doLabel_cabd_all[detect_result[0]] = True
    d = {'index': np.arange(1, len(data) + 1), 'value': data, 'label': label_data_cabd_all,
         'truth': list_truth, 'doLabel': doLabel_cabd_all, 'anomaly': anomaly_points_data}
    df = pd.DataFrame(data=d)
    df = df[['index', 'value', 'label', 'truth', 'doLabel', 'anomaly']]
    df.to_csv(file_path + "/" + DATA_FILE + "_label_detection.csv", index=False)

###############################################__LABEL BASED ON RANDOM _____#########################################################
    label_data_random = data.copy();
    doLabel_random = np.full((1, len(data)), False)[0]
    ramdom_list = np.random.randint(4, len(data), int(len(data) * 0.2))
    for index, value in enumerate(ramdom_list):
        label_data_random[value] = list_truth[value]
        doLabel_random[value] = True
    d = {'index': np.arange(1, len(data) + 1), 'value': data, 'label': label_data_random,
         'truth': list_truth, 'doLabel': doLabel_random, 'anomaly': anomaly_points_data}
    df = pd.DataFrame(data=d)
    df = df[['index', 'value', 'label', 'truth', 'doLabel', 'anomaly']]
    df.to_csv(file_path + "/" + DATA_FILE + "_label_random.csv", index=False)


def display_performance(X_data, array_evaluation_result_myAL, dict_result_analytic, dict_score, ground_anomaly_list,
                        ground_change_point_list, learner):
    result_anomaly_list = []
    result_changepoint_list = []
    examing_point = list(dict_score.keys())
    model_predict_result = learner.predict(X_data)
    result_all = []

    for tempt_i, tempt_value in enumerate(model_predict_result):
        if tempt_value == 3:
            result_changepoint_list.append(examing_point[tempt_i])
            result_all.append([examing_point[tempt_i],tempt_value, max(learner.predict_proba([X_data[tempt_i]])[0])])
    for tempt_i, tempt_value in enumerate(model_predict_result):
        if examing_point[tempt_i] not in result_changepoint_list:
            if tempt_value == 1:
                result_anomaly_list.append(examing_point[tempt_i])
                result_all.append([examing_point[tempt_i], tempt_value, max(learner.predict_proba([X_data[tempt_i]])[0])])
                if len(dict_result_analytic[examing_point[tempt_i]][2]) > 1:
                    result_anomaly_list.extend(
                        [z for z in dict_result_analytic[examing_point[tempt_i]][2]])
            if tempt_value == 2:
                # result_anomaly_list.append(examing_point[tempt_i])
                result_all.append([[z for z in dict_result_analytic[examing_point[tempt_i]][2]], tempt_value, max(learner.predict_proba([X_data[tempt_i]])[0])])
                result_anomaly_list.extend(
                    [z for z in dict_result_analytic[examing_point[tempt_i]][2]])
    # Calculate the correctness.
    temp_recal_value = 100 * len(set(ground_anomaly_list).intersection(set(result_anomaly_list))) / len(
        set(ground_anomaly_list)) if len(set(ground_anomaly_list)) != 0 else 0
    temp_precision_value = 100 * len(set(ground_anomaly_list).intersection(set(result_anomaly_list))) / len(
        set(result_anomaly_list)) if len(set(result_anomaly_list)) != 0 else 0
    temp_recal_value_changePoint = 100 * len(
        set(ground_change_point_list).intersection(set(result_changepoint_list))) / len(
        set(ground_change_point_list)) if len(set(ground_change_point_list)) != 0 else 0
    temp_precision_value_changePoint = 100 * len(
        set(ground_change_point_list).intersection(set(result_changepoint_list))) / len(
        set(result_changepoint_list)) if len(
        set(result_changepoint_list)) != 0 else 0
    array_evaluation_result_myAL['recall_anomaly'].append(temp_recal_value)
    array_evaluation_result_myAL['precision_anomaly'].append(temp_precision_value)
    array_evaluation_result_myAL['recall_changePoint'].append(temp_recal_value_changePoint)
    array_evaluation_result_myAL['precision_changePoint'].append(temp_precision_value_changePoint)
    print("Anomaly Detection Precision: {}".format(temp_precision_value))
    print("Anomaly Detection Recall: {}".format(temp_recal_value))
    print("Change Point Detection Precision: {}".format(temp_precision_value_changePoint))
    print("Change Point Detection Recall: {}".format(temp_recal_value_changePoint))
    return [temp_precision_value,temp_recal_value, temp_precision_value_changePoint, temp_recal_value_changePoint,result_all]


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None

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
