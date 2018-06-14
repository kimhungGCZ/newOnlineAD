import urllib.request as urllib
import json
import numpy as np
import pandas as pd
import time
import datetime
import warnings
import matplotlib.pyplot as plt
import scripts.data_generation as data_generation
from random import randint
import os

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


def getGCZDataFrame(DATA_FILE, AN_per, CP_per):

    # request_URL = "https://server.humm-box.com/api/devices/1B3AEA/fastmeasures?fields=[content_volume]"
    # request = urllib.Request(request_URL)
    # request.add_header("Authorization",
    #                    "bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2h1bW0tc2VydmVyLmV1LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNTQ5MjcyODgyOTQ0NjU4MzExNiIsImF1ZCI6IkxMSWVDYXpIVEpTOG9kVW1kaHJHMmVuV3dQaW5yajUxIiwiaWF0IjoxNTIyOTIyMDkzLCJleHAiOjE1MjY1MjIwOTN9.tNhBbGvdZQdUq0WKLz6fN5QKspLxijuENiFBztPqwU4")
    # webURL = urllib.urlopen(request)
    # tem_data = json.load(webURL)
    # tem_data.sort(key=lambda x: x[0])
    # data = [i[1] for index, i in enumerate(tem_data) if i[1] != None] #and index < 2000]#and i[1] > 0]#
    # #data = [i for index, i in enumerate(data) if index < 350 or index > 700]#and i[1] > 0]#

    # df = pd.read_csv("./active_result/data_1B3B8D.csv")
    # data = df['value'].values
    # plt.plot(data)
    # plt.show()
    ############################### GENERATE THE DATA ################################
    # file_path = "./active_result/all/" + DATA_FILE + "/" + DATA_FILE  + ".csv"
    # directory = os.path.dirname(file_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # raw_data_generation = data_generation.generate_symetric_dataset_noise(AN_per, CP_per)
    # data = raw_data_generation[0]
    # list_change_points = raw_data_generation[1]
    # list_anonaly_points = raw_data_generation[2]
    # list_anomaly_pattern = raw_data_generation[3]
    # print("N0 Change point: {}".format(len(list_change_points)))
    # print("N0 Anomaly point: {}".format(len(list_anonaly_points)))
    # print("N0 Anomaly Pattern point: {}".format(len(list_anomaly_pattern)))
    #
    # change_points_data = np.zeros(len(data))
    # change_points_data[list_change_points[0:-1]] = 1
    #
    # anomaly_points_data = np.zeros(len(data))
    # anomaly_points_data[list_anonaly_points] = 1
    #
    # anomaly_pattern_points_data = np.zeros(len(data))
    # for i in list_anomaly_pattern:
    #     anomaly_pattern_points_data[i] = 1
    #
    # ts = time.time()
    # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    # time_array = [datetime.datetime.fromtimestamp(ts - 10000 * i).strftime('%Y-%m-%d %H:%M:%S') for i in data]
    # d = {'timestamp': time_array, 'value': data, 'change_point': change_points_data,
    #      'anomaly_point': anomaly_points_data, 'anomaly_pattern': anomaly_pattern_points_data}
    # df = pd.DataFrame(data=d)
    # df.to_csv(file_path, index=False);
    ############################## LOAD FROM CSV ###################################
    # file_path = "./active_result/yahoo/"+DATA_FILE+"/" + DATA_FILE + ".csv"
    # directory = os.path.dirname(file_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # df = pd.read_csv("./active_result/test/A1Benchmark/" + DATA_FILE + ".csv")
    # anomaly_points_data = df['is_anomaly'].values
    #
    # value = df['value'].values
    # anomaly_pattern_points_data = np.zeros(len(value))
    # change_points_data = np.zeros(len(value))
    # ts = time.time()
    # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    # time_array = [datetime.datetime.fromtimestamp(ts - 10000 * i).strftime('%Y-%m-%d %H:%M:%S') for i,v in enumerate(value)]
    # d = {'timestamp': time_array, 'value': value, 'change_point': change_points_data,
    #      'anomaly_point': anomaly_points_data, 'anomaly_pattern': anomaly_pattern_points_data}
    # df_new = pd.DataFrame(data=d)
    # df_new.to_csv(file_path, index=False);

    ############################## LOAD FROM  ###################################
    df = pd.read_csv("./active_result/all/" + DATA_FILE + "/" + DATA_FILE + ".csv")

    # webURL.close()
    return df

# request = urllib2.Request("https://server.humm-box.com/api/devices/2004DF/fastmeasures?fields=[content_volume]")
# request.add_header("Authorization",
#                   "bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2h1bW0tc2VydmVyLmV1LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNTQ5MjcyODgyOTQ0NjU4MzExNiIsImF1ZCI6IkxMSWVDYXpIVEpTOG9kVW1kaHJHMmVuV3dQaW5yajUxIiwiZXhwIjoxNTEwMDI2NDU1LCJpYXQiOjE1MDY0MjY0NTV9.gma7GCb2-dYiMnJkeapyrd2Y_xhk0Wk_14zS49Yk7Pc")
# result = urllib2.urlopen(request)
# tem_data = json.load(result.fp)
# tem_data.sort(key=lambda x: x[0])
# data = [i[1] for index, i in enumerate(tem_data) if i[1] != None and index < 2000]
# plt.plot(data)
# plt.show()
# ts = time.time()
# st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
# time_array = [datetime.datetime.fromtimestamp(ts - 10000 * i).strftime('%Y-%m-%d %H:%M:%S') for i in data]
# d = {'timestamp': time_array, 'value': data}
# df = pd.DataFrame(data=d)
# df.to_csv("./data/realKnownCause/data_2004DF.csv", index=False);
# result.close()
