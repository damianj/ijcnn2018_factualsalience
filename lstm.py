import json
import pickle
import glob
from datetime import datetime
from math import floor
from pathlib import Path

import numpy
from keras.regularizers import l1, l2
from keras.layers import Dense, Bidirectional, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.metrics import mse, cosine, categorical_accuracy
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

import data_helper


class ClaimBusterRNN:
    def __init__(self, data_file=None, rand_shuffle=False, rng_state_file=None, to_two_class=False):
        self.to_two_class = to_two_class
        self.data_file = data_file
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.labels = None
        self.rand_shuffle = rand_shuffle
        self.model = None
        self.scores = None
        self.path = None
        self.rng_state_file = rng_state_file
        self.rng_state = None

    def create_dirs(self, folder_name=None):
        self.path = "{0}".format(folder_name or datetime.now().strftime("m%m-d%d-y%Y--h%H_m%M_s%S_µs%f"))
        Path(self.path).mkdir(parents=True, exist_ok=True)
        print("Directories created: {0}".format(self.path))

    def load_data(self, file=None):
        self.x, self.y, self.labels = data_helper.load_data_and_labels(file or self.data_file, self.to_two_class)

        if self.rand_shuffle:
            if self.rng_state_file:
                with open(self.rng_state_file, mode="rb") as rng_state_pkl:
                    self.rng_state = pickle.load(rng_state_pkl)
            else:
                self.rng_state = numpy.random.get_state()
                with open("{0}/rng.state".format(self.path), mode="wb") as rng_state_pkl:
                    pickle.dump(self.rng_state, rng_state_pkl)

            numpy.random.set_state(self.rng_state)
            shuffled_indices = numpy.random.permutation(numpy.arange(len(self.y)))
            self.x = self.x[shuffled_indices]
            self.y = self.y[shuffled_indices]
        print("Data loaded...")

    def partition_data(self, test_size=0.20, random_state=42):
        shuffle = False
        stratify = None

        if not self.rand_shuffle:
            shuffle = True
            stratify = self.y

        if self.rng_state_file:
            with open(self.rng_state_file, mode="rb") as rng_state_pkl:
                self.rng_state = pickle.load(rng_state_pkl)
        else:
            self.rng_state = numpy.random.get_state()
            with open("{0}/rng.state".format(self.path), mode="wb") as rng_state_pkl:
                pickle.dump(self.rng_state, rng_state_pkl)

        numpy.random.set_state(self.rng_state)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=shuffle, stratify=stratify)
        print("Data partitioned...")

    def create_model(self, filter_sizes=(2, 3, 4), activation="tanh", loss="mean_squared_error", optimizer="nadam"):
        self.model = Sequential()
        self.model.add(Dense(len(self.x_train[0]), input_shape=(len(self.x_train[0]), len(self.x_train[0][0]))))

        for filter_len in filter_sizes:
            self.model.add(Conv1D(filters=128, kernel_size=filter_len, padding='same'))

        self.model.add(Bidirectional(LSTM(128, unit_forget_bias=True)))
        self.model.add(Dense(len(self.y[0]), activation=activation, kernel_regularizer=l2(0.005),
                             activity_regularizer=l1(0.001)))

        self.model.compile(loss=loss, optimizer=optimizer, metrics=[cosine, mse, categorical_accuracy])
        print("Model created...")

    def fit_model(self, epochs=30, batch_size=128):
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def visualize(self, f_name="model"):
        if self.model:
            plot_model(self.model, to_file="{0}.png".format(f_name))
        else:
            print("Error: No model defined!")

    def evaluate_model(self):
        self.scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)

        with open("{0}/summary.txt".format(self.path), mode="w") as summary:
            with open("{0}/results.json".format(self.path), mode="w") as results_json:
                ground_truth = self.multi_2_single_label(self.y_test)
                predictions = self.multi_2_single_label(self.cont_2_disc(self.model.predict(self.x_test)))
                c_matrix = confusion_matrix(ground_truth, predictions).tolist()
                results = {"accuracy": self.scores[3] * 100, "conf_matrix": c_matrix, "labels": ["CFS", "NFS", "UFS"]}

                summary.write("Accuracy: {0:.3f}%\n\n".format(self.scores[3] * 100))
                summary.write("{0}".format(self.pprint_c_matrix(c_matrix, ["CFS", "NFS", "UFS"])))
                summary.write("\n\nMeasure\t\t\tCFS\t\t\tNFS\t\t\tUFS\t\tWeighted Average\n")

                cfs, nfs, ufs = precision_score(ground_truth, predictions, average=None)
                w_a = precision_score(ground_truth, predictions, average="weighted")
                results["precision"] = {"csf": cfs, "nfs": nfs, "ufs": ufs, "weighted_average": w_a}
                summary.write("Precision\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\t\t{3:.5f}\n".format(cfs, nfs, ufs, w_a))

                cfs, nfs, ufs = recall_score(ground_truth, predictions, average=None)
                w_a = recall_score(ground_truth, predictions, average="weighted")
                results["recall"] = {"csf": cfs, "nfs": nfs, "ufs": ufs, "weighted_average": w_a}
                summary.write("Recall\t\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\t\t{3:.5f}\n".format(cfs, nfs, ufs, w_a))

                cfs, nfs, ufs = f1_score(ground_truth, predictions, average=None)
                w_a = f1_score(ground_truth, predictions, average="weighted")
                results["f1_score"] = {"csf": cfs, "nfs": nfs, "ufs": ufs, "weighted_average": w_a}
                summary.write("F1-Score\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\t\t{3:.5f}\n".format(cfs, nfs, ufs, w_a))
                json.dump(results, results_json)

    def evaluate_binary_model(self):
        self.scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)

        with open("{0}/summary.txt".format(self.path), mode="w") as summary:
            with open("{0}/results.json".format(self.path), mode="w") as results_json:
                ground_truth = self.multi_2_single_label(self.y_test)
                predictions = self.multi_2_single_label(self.cont_2_disc(self.model.predict(self.x_test)))
                c_matrix = confusion_matrix(ground_truth, predictions).tolist()
                results = {"accuracy": self.scores[3] * 100, "conf_matrix": c_matrix, "labels": ["CFS", "N-UFS"]}

                summary.write("Accuracy: {0:.3f}%\n\n".format(self.scores[3] * 100))
                summary.write("{0}".format(self.pprint_c_matrix(c_matrix, ["CFS", "N-UFS"])))
                summary.write("\n\nMeasure\t\t\tCFS\t\t\tN-UFS\t\tWeighted Average\n")

                cfs, nufs = precision_score(ground_truth, predictions, average=None)
                w_a = precision_score(ground_truth, predictions, average="weighted")
                results["precision"] = {"csf": cfs, "nfs": nufs, "weighted_average": w_a}
                summary.write("Precision\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\n".format(cfs, nufs, w_a))

                cfs, nufs = recall_score(ground_truth, predictions, average=None)
                w_a = recall_score(ground_truth, predictions, average="weighted")
                results["recall"] = {"csf": cfs, "nfs": nufs, "weighted_average": w_a}
                summary.write("Recall\t\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\n".format(cfs, nufs, w_a))

                cfs, nufs = f1_score(ground_truth, predictions, average=None)
                w_a = f1_score(ground_truth, predictions, average="weighted")
                results["f1_score"] = {"csf": cfs, "nfs": nufs, "weighted_average": w_a}
                summary.write("F1-Score\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\n".format(cfs, nufs, w_a))
                json.dump(results, results_json)

    def save_model(self, dir_name="data", weights_name="model.weights"):
        Path("{0}/{1}/".format(self.path, dir_name)).mkdir(parents=True, exist_ok=True)

        self.model.save("{0}/{1}/model.h5".format(self.path, dir_name))
        self.model.save_weights("{0}/{1}/{2}.h5".format(self.path, dir_name, weights_name))

        with open("{0}/{1}/model.json".format(self.path, dir_name), mode="w") as json_file:
            json.dump(self.model.to_json(), json_file)

        """""
        with open("{0}/{1}/x.pkl".format(self.path, dir_name), mode="wb") as x_pkl:
            pickle.dump(self.x.tolist(), x_pkl)

        with open("{0}/{1}/x_train.pkl".format(self.path, dir_name), mode="wb") as x_train_pkl:
            pickle.dump(self.x_train.tolist(), x_train_pkl)

        with open("{0}/{1}/x_test.pkl".format(self.path, dir_name), mode="wb") as x_test_pkl:
            pickle.dump(self.x_test.tolist(), x_test_pkl)

        with open("{0}/{1}/y.pkl".format(self.path, dir_name), mode="wb") as y_pkl:
            pickle.dump(self.y.tolist(), y_pkl)

        with open("{0}/{1}/y_train.pkl".format(self.path, dir_name), mode="wb") as y_train_pkl:
            pickle.dump(self.y_train.tolist(), y_train_pkl)

        with open("{0}/{1}/y_test.pkl".format(self.path, dir_name), mode="wb") as y_test_pkl:
            pickle.dump(self.y_test.tolist(), y_test_pkl)
        """""

    @staticmethod
    def cont_2_disc(arr):
        discrete_arr = arr
        for inner_index, item in enumerate(discrete_arr):
            outer_index = item.argmax()
            discrete_arr[inner_index] *= 0
            discrete_arr[inner_index][outer_index] = 1

        return discrete_arr

    @staticmethod
    def multi_2_single_label(arr):
        single_label_arr = []
        for item in arr:
            single_label_arr.append(item.argmax())

        return numpy.asarray(single_label_arr)

    @staticmethod
    def pprint_c_matrix(matrix, labels=(1, 2, 3)):
        pprint_str = "\t\t\t\t\t{0: ^24}\n\t\t\t\t\t".format("Predictions")

        for label in labels:
            pprint_str += "{0: ^7}".format(label)

        pprint_str += "\n"
        try:
            for row_num, row in enumerate(matrix, start=1):
                if row_num == floor((len(matrix) + 1) / 2.0):
                    pprint_str += "Actual\t{0: ^10}\t".format(labels[row_num - 1])
                else:
                    pprint_str += "\t\t{0: ^10}\t".format(labels[row_num - 1])
                for num in row:
                    pprint_str += "{0: ^7}".format(num)
                pprint_str += "\n"
        except IndexError:
            pprint_str = "Label array must be the same length as the number of rows/columns in the confusion matrix " \
                         "for it to be printed successfully."

        return pprint_str


classes = {"two_class": True, "three_class": False}
model_kwargs_list = [{"activation": "softmax", "loss": "logcosh", "optimizer": "adam"},
                     {"activation": "softmax", "loss": "categorical_crossentropy", "optimizer": "adam"}]


num_classes = "three_class"
model_kwargs = model_kwargs_list[1]
base_path = ""

if model_kwargs["loss"] == "logcosh":
    base_path += "regression_models"
else:
    base_path += "classifier_models"

run_path = "parse_tree_runs/"

claim_buster_rnn = ClaimBusterRNN(None, to_two_class=classes[num_classes])
try:
    claim_buster_rnn.create_dirs(run_path)
    claim_buster_rnn.load_data()
    claim_buster_rnn.partition_data()
    claim_buster_rnn.create_model(**model_kwargs)
    claim_buster_rnn.fit_model(epochs=15, batch_size=128)
    claim_buster_rnn.save_model()
    if classes[num_classes]:
        claim_buster_rnn.evaluate_binary_model()
    else:
        claim_buster_rnn.evaluate_model()
except Exception as e:
    with open("{0}/fail.log".format(claim_buster_rnn.path), mode="w") as fail_txt:
        fail_txt.write("Error:\n{0}".format(e))
