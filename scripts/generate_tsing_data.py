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

def generate_tsing_data_format():
    DATA_FILE = "test_001_001"
    AN_per = 0.01
    CP_per = 0.01

    file_path = "./active_result/tsing/" + DATA_FILE + "/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    raw_data_generation = data_generation.generate_symetric_dataset_noise(AN_per, CP_per)
    data = raw_data_generation[0]
    list_change_points = raw_data_generation[1]
    list_anonaly_points = raw_data_generation[2]
    list_anomaly_pattern = raw_data_generation[3]
    list_truth = raw_data_generation[4]

    change_points_data = np.zeros(len(data))
    change_points_data[list_change_points[0:-1]] = 1

    anomaly_points_data = np.zeros(len(data))
    anomaly_points_data[list_anonaly_points] = 1

    anomaly_points_data[list_anomaly_pattern] = 1

    label_data = data.copy();
    doLabel = np.full((1, len(data)), False)[0]

    d = {'index': np.arange(1, len(data) + 1), 'value': data, 'label': label_data,
         'truth': list_truth, 'doLabel': doLabel, 'anomaly': anomaly_points_data}
    df = pd.DataFrame(data=d)
    df = df[['index', 'value', 'label', 'truth', 'doLabel', 'anomaly']]
    df.to_csv(file_path + "/" + DATA_FILE + "_original.csv", index=False);

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


def format_real_dataset(DATA_FILE):
    df = pd.read_csv("./active_result/tsing/" + DATA_FILE + ".csv")
    random_label = np.random.randint(0,len(df),np.int(0.2*len(df)))
    for i in random_label:
        df['label'][i] = df['truth'][i];
        df['doLabel'][i] = True
    df.to_csv('./active_result/'+DATA_FILE+".csv", index=False);
