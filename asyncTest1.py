from random import randint
from scripts.detect_peaks import detect_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg

def generate_symetric_dataset(pattern_number = 5):

    initial_index = 0;
    signal = []
    list_change_points = []
    list_anonaly_points = []
    list_anomaly_pattern = []
    while initial_index < pattern_number:
        pattern_top = randint(1000, 1000)
        pattern_bot = randint(200, 200)
        pattern_size = randint(1400, 1400)
        signal_index = len(signal)

        Fs = 1400
        f = 1
        x = np.arange(pattern_size)

        #pure = np.linspace(pattern_top, pattern_bot, pattern_size)
        pure = pattern_top*np.sin(2 * np.pi * f * x / Fs)
        pure = pattern_top* sg.square(2 *np.pi * f *x / Fs )
        noise = np.random.normal(-50, 50, pure.shape)
        temp_signal = pure + noise
        anomaly_flag = np.random.choice([0, 1], 1, replace=False, p=[0.8, 0.2])
        if anomaly_flag == 1:
            anomaly_single = np.random.randint(pattern_top - 100, high=pattern_top, size=randint(2, 6))
            anomaly_final = np.zeros(pattern_size)
            for i in anomaly_single:
                temp_anomaly_pos = randint(0, pattern_size - 1)
                anomaly_final[temp_anomaly_pos] = randint(np.int(pattern_top - temp_signal[temp_anomaly_pos] - 100),
                                                          np.int(pattern_top - temp_signal[temp_anomaly_pos]))
                list_anonaly_points.append(temp_anomaly_pos + signal_index)
            temp_signal = temp_signal + anomaly_final
        anomaly_flag = np.random.choice([0, 1], 1, replace=False, p=[0.8, 0.2])
        if anomaly_flag == 1:
            anomaly_pattern_size = randint(5, 10)
            anomaly_final = np.zeros(pattern_size)
            temp_anomaly_pos = randint(np.int(pattern_size/2), pattern_size - 1)
            anomaly_final[temp_anomaly_pos:temp_anomaly_pos + anomaly_pattern_size] = np.linspace(
                np.int(pattern_top - temp_signal[temp_anomaly_pos]),
                np.int(pattern_top - temp_signal[temp_anomaly_pos] - 20), anomaly_pattern_size)
            temp_signal = temp_signal + anomaly_final
            list_anomaly_pattern.append([i + signal_index for i in range(temp_anomaly_pos,temp_anomaly_pos + anomaly_pattern_size)])
        signal.extend(temp_signal)

        list_change_points.append(signal_index + pattern_size)
        initial_index = initial_index + 1
    return [signal,list_change_points, list_anonaly_points, list_anomaly_pattern]

plt.plot(generate_symetric_dataset(10)[0])
plt.show()