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
        # result_anomaly_list.extend(dict_result_analytic[j[0]][2])
        result_anomaly_list.extend([z for z in dict_result_analytic[j[0]][2] if z not in list_uncertainty_point])
    for j in list_changed_point:
        result_changepoint_list.append(j[0])

    temp_recal_value = 100 * len(set(ground_anomaly_list).intersection(set(result_anomaly_list))) / len(
        ground_anomaly_list) if len(ground_anomaly_list) != 0 else 0
    temp_precision_value = 100 * len(set(ground_anomaly_list).intersection(set(result_anomaly_list))) / len(
        result_anomaly_list) if len(result_anomaly_list) != 0 else 0

    temp_recal_value_changePoint = 100 * len(
        set(ground_change_point_list).intersection(set(result_changepoint_list))) / len(
        ground_change_point_list) if len(ground_change_point_list) != 0 else 0
    temp_precision_value_changePoint = 100 * len(
        set(ground_change_point_list).intersection(set(result_changepoint_list))) / len(result_changepoint_list) if len(
        result_changepoint_list) != 0 else 0

    array_evaluation_result['recall_anomaly'].append(temp_recal_value)
    array_evaluation_result['precision_anomaly'].append(temp_precision_value)
    array_evaluation_result['recall_changePoint'].append(temp_recal_value_changePoint)
    array_evaluation_result['precision_changePoint'].append(temp_precision_value_changePoint)

    outF = open(file_path_chart + ".txt", "a")
    outF.write("Round " + str(round_index_value))
    outF.write("\n")
    outF.write("Anomaly Detection Correctness: " + str(temp_precision_value))
    outF.write("\n")
    outF.write("Change Point Detection Correctness: " + str())
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