import os
import sys
from time import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
import scripts.obtain_data as data_engine
import scripts.data_generation as data_generation
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
import pandas as pd


# Define the number of inliers and outliers
outliers_fraction = 0.05
clusters_separation = [0]

# # Compare given detectors under given settings
# # Initialize the data
# xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
# n_inliers = int((1. - outliers_fraction) * n_samples)
# n_outliers = int(outliers_fraction * n_samples)
# ground_truth = np.zeros(n_samples, dtype=int)
# ground_truth[-n_outliers:] = 1
#
#
#
# # Show the statics of the data
# print('Number of inliers: %i' % n_inliers)
# print('Number of outliers: %i' % n_outliers)
# print('Ground truth shape is {shape}. Outlier are 1 and inliers are 0.\n'.format(shape=ground_truth.shape))
# print(ground_truth)

def evaluate_function(DATA_NAME):
    random_state = False
    # Define nine outlier detection tools to be compared
    classifiers = {'Angle-based Outlier Detector (ABOD)':
                       ABOD(n_neighbors=10,
                            contamination=outliers_fraction),
                   'Cluster-based Local Outlier Factor (CBLOF)':
                       CBLOF(contamination=outliers_fraction,
                             check_estimator=False, random_state=random_state),
                   'Feature Bagging':
                       FeatureBagging(LOF(n_neighbors=35),
                                      contamination=outliers_fraction,
                                      check_estimator=False,
                                      random_state=random_state),
                   'Histogram-base Outlier Detection (HBOS)': HBOS(
                       contamination=outliers_fraction),
                   'Isolation Forest': IForest(contamination=outliers_fraction,
                                               random_state=random_state),
                   'K Nearest Neighbors (KNN)': KNN(
                       contamination=outliers_fraction),
                   'Average KNN': KNN(method='mean',
                                      contamination=outliers_fraction),
                   'Median KNN': KNN(method='median',
                                     contamination=outliers_fraction),
                   'Local Outlier Factor (LOF)':
                       LOF(n_neighbors=35, contamination=outliers_fraction),
                   'Minimum Covariance Determinant (MCD)': MCD(
                       contamination=outliers_fraction, random_state=random_state),
                   'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction,
                                                  random_state=random_state),
                   'Principal Component Analysis (PCA)': PCA(
                       contamination=outliers_fraction, random_state=random_state),
                   "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
                   "One-Class SVM": svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                                    gamma=0.1)
                   }
    for i, offset in enumerate(clusters_separation):
        np.random.seed(42)
        # Data generation
        data_frame = data_engine.getGCZDataFrame(DATA_NAME)
        X_all = np.array(list(map(lambda x: [x, data_frame.values[x][4]], np.arange(len(data_frame.values)))))
        X_train = np.array([[x, data_frame.values[x][4]] for x in np.arange(len(data_frame.values)) if
                            data_frame.values[x][0] != 1 or data_frame.values[x][1] != 1])
        X_train_ano = np.array([[x, data_frame.values[x][4]] for x in np.arange(len(data_frame.values)) if
                                data_frame.values[x][0] == 1 or data_frame.values[x][1] == 1])

        ground_anomaly_list = [index for index, value in enumerate(data_frame['anomaly_pattern'].values) if value == 1]
        ground_anomaly_list.extend(
            [index for index, value in enumerate(data_frame['anomaly_point'].values) if value == 1])
        ground_truth = ground_anomaly_list

        for i, (clf_name, clf) in enumerate(classifiers.items()):
            print(i + 1, 'fitting', clf_name)
            # fit the data and tag outliers
            clf.fit(X_all)
            y_pred = clf.predict(X_all)

            final_result = [index for index, value in enumerate(y_pred) if value == 1]

            temp_recal_value = 100 * len(set(ground_truth).intersection(set(final_result))) / len(
                set(ground_truth)) if len(set(ground_truth)) != 0 else 0
            temp_precision_value = 100 * len(set(ground_truth).intersection(set(final_result))) / len(
                set(final_result)) if len(set(final_result)) != 0 else 0
            print("Precision: ", temp_precision_value, " Recal: ", temp_recal_value)
            # plot the levels lines and the points
            ################## WRITE TO CSV FILE ################################
            try:
                df_final_result = pd.read_csv(os.path.normpath(
                    'D:/Google Drive/13. These cifre/Data Cleaning/workspace/SVADS/' + DATA_NAME + '.csv'))

                df_final_result = df_final_result.append({'ALname': clf_name,
                                                          'precision': temp_precision_value,
                                                          'recall': temp_recal_value,
                                                          }, ignore_index=True)
                df_final_result.to_csv(os.path.normpath(
                    'D:/Google Drive/13. These cifre/Data Cleaning/workspace/SVADS/' + DATA_NAME + '.csv'),
                    index=False);
            except FileNotFoundError:

                df_final_result = pd.DataFrame([[clf_name,
                                                 temp_precision_value,
                                                 temp_recal_value
                                                 ]], columns=['ALname', 'precision', 'recall'])
                df_final_result.to_csv(os.path.normpath(
                    'D:/Google Drive/13. These cifre/Data Cleaning/workspace/SVADS/' + DATA_NAME + '.csv'),
                    index=False);


if __name__== "__main__":
    base_name = "real_"
    data_array = []  # real
    #data_array = ["example 320387", "example 346500", "example 533964","example 645266"] # SAW
    #data_array = ["example 624622", "example 798717", "example 513024"] # SIN
    #data_array = ["example 348800", "example 387713", "example 692083", "example 961480", "example 989638"] # SQUARE
    #data_array = ["real_8"] # real
    # for i in range(1,50):
    #     if i not in [1,7,10,20]:
    #         data_array.append(base_name + str(i))
    AL_coup = [[0.01, 85], [0.05, 80], [0.1, 75], [0.15, 70], [0.2, 65]]
    CP_coup = [0.01, 0.02, 0.05, 0.1, 0.2]
    #CP_coup = [0.2]
    # CP_coup = [0.01, 0.02, 0.05, 0.1, 0.15]
    # AL_coup = [[0.1,75],[0.15,70],[0.2,65]]
    # AL_coup = [[0.01,60]]
    for CP_value in CP_coup:
        for run_value in AL_coup:
            data_name = ("test_" + str(run_value[0]) + "_" + str(CP_value)).replace(".", "")
            data_array.append(data_name)


    #data_array = ["test_001_01","test_001_01"]
    K_value_array = np.arange(5,100,5)
    for data in data_array:
        print("############################# START AT DATASET: {} ##########################################".format(data))
        try:
            detect_final_result = evaluate_function(data);
        except:
            print("ERROR at {}".format(data))
