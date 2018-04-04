import urllib.request as urllib
import json
import numpy as np
import pandas as pd
import time
import datetime
import warnings
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


def getGCZDataFrame(DATA_FILE):
    # request_URL = "https://server.humm-box.com/api/devices/" + deviceID + "/fastmeasures?fields=[content_volume]"
    # request = urllib.Request(request_URL)
    # request.add_header("Authorization",
    #                    "bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2h1bW0tc2VydmVyLmV1LmF1dGgwLmNvbS8iLCJzdWIiOiJhdXRoMHw1ODM5YmMxMjAwNjk2NDA3MDdmMGQ1ZjkiLCJhdWQiOiJMTEllQ2F6SFRKUzhvZFVtZGhyRzJlbld3UGlucmo1MSIsImlhdCI6MTUxODE3NjQ1MywiZXhwIjoxNTIxNzc2NDUzfQ.i23hAZKBlmoOkt9XorTsdUWLB6GRzOA4Kc_zUvwgTz4")
    # webURL = urllib.urlopen(request)
    # tem_data = json.load(webURL)
    # tem_data.sort(key=lambda x: x[0])
    # data = [i[1] for index, i in enumerate(tem_data) if i[1] != None and index < 2000]#and i[1] > 0]#
    # data = [i for index, i in enumerate(data) if index < 350 or index > 700]#and i[1] > 0]#
    # plt.plot(data)
    # plt.show()

    ############################### GENERATE THE DATA ################################
    # file_path = "./active_result/test/" + DATA_FILE + "/" + DATA_FILE + ".csv"
    # directory = os.path.dirname(file_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # raw_data_generation = data_generation.generate_symetric_dataset(pattern_number=10)
    # data = raw_data_generation[0]
    # list_change_points = raw_data_generation[1]
    # list_anonaly_points = raw_data_generation[2]
    # list_anomaly_pattern = raw_data_generation[3]
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
    df = pd.read_csv("./active_result/all/"+ DATA_FILE + "/" + DATA_FILE + ".csv")

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
