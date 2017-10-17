import urllib.request as urllib
import json
import numpy as np
import pandas as pd
import time
import datetime
import warnings

warnings.simplefilter('ignore')


def getCSVData(dataPath):
    try:
        data = pd.read_csv(dataPath)
    except IOError("Invalid path to data file."):
        return
    return data


def getGCZDataFrame(deviceID):
    request_URL = "https://server.humm-box.com/api/devices/" + deviceID + "/fastmeasures?fields=[content_volume]"
    request = urllib.Request(request_URL)
    request.add_header("Authorization",
                       "bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2h1bW0tc2VydmVyLmV1LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNTQ5MjcyODgyOTQ0NjU4MzExNiIsImF1ZCI6IkxMSWVDYXpIVEpTOG9kVW1kaHJHMmVuV3dQaW5yajUxIiwiZXhwIjoxNTA5MDg0ODc4LCJpYXQiOjE1MDU0ODQ4Nzh9.xbEiLpIuorLYGT4UJZ6bKrmy9f8uugnwNTe5YyuzJko")
    webURL = urllib.urlopen(request)
    tem_data = json.load(webURL)
    tem_data.sort(key=lambda x: x[0])
    data = [i[1] for index, i in enumerate(tem_data) if i[1] != None]
    #plt.plot(data)
    #plt.show()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    time_array = [datetime.datetime.fromtimestamp(ts - 10000 * i).strftime('%Y-%m-%d %H:%M:%S') for i in data]
    d = {'timestamp': time_array, 'value': data}
    df = pd.DataFrame(data=d)
    webURL.close()
    return df


#request = urllib2.Request("https://server.humm-box.com/api/devices/2004DF/fastmeasures?fields=[content_volume]")
#request.add_header("Authorization",
#                   "bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2h1bW0tc2VydmVyLmV1LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNTQ5MjcyODgyOTQ0NjU4MzExNiIsImF1ZCI6IkxMSWVDYXpIVEpTOG9kVW1kaHJHMmVuV3dQaW5yajUxIiwiZXhwIjoxNTEwMDI2NDU1LCJpYXQiOjE1MDY0MjY0NTV9.gma7GCb2-dYiMnJkeapyrd2Y_xhk0Wk_14zS49Yk7Pc")
#result = urllib2.urlopen(request)
#tem_data = json.load(result.fp)
#tem_data.sort(key=lambda x: x[0])
#data = [i[1] for index, i in enumerate(tem_data) if i[1] != None and index < 2000]
#plt.plot(data)
#plt.show()
#ts = time.time()
#st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#time_array = [datetime.datetime.fromtimestamp(ts - 10000 * i).strftime('%Y-%m-%d %H:%M:%S') for i in data]
#d = {'timestamp': time_array, 'value': data}
#df = pd.DataFrame(data=d)
#df.to_csv("./data/realKnownCause/data_2004DF.csv", index=False);
#result.close()
