def online_anomaly_detection_optimize(result_dta, raw_dta, alpha, DATA_FILE):
    median_sec_der = np.mean(result_dta['value'])
    std_sec_der = np.std(result_dta['value'])
    dta_full = result_dta
    dta_full.value.index = result_dta.timestamp

    std_anomaly_set = np.std(result_dta['anomaly_score'])
    np.argsort(result_dta['anomaly_score'])

    anomaly_index = np.array([i for i, value in enumerate(result_dta['anomaly_score']) if value > 3 * std_anomaly_set])

    limit_size = int(1 / alpha)

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
                list(raw_dta.value.values[
                         list(z for z in range(int(min(detect_pattern) - 3), int(min(detect_pattern))) if
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