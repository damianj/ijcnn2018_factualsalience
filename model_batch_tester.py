import data_helper
import numpy
import glob
import json
from math import floor
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def cont_2_disc(arr):
    discrete_arr = arr
    for inner_index, item in enumerate(discrete_arr):
        outer_index = item.argmax()
        discrete_arr[inner_index] *= 0
        discrete_arr[inner_index][outer_index] = 1

    return discrete_arr


def multi_2_single_label(arr):
    single_label_arr = []
    for item in arr:
        single_label_arr.append(item.argmax())

    return numpy.asarray(single_label_arr)


def pprint_c_matrix(matrix, lbl=(1, 2, 3)):
    pprint_str = "\t\t\t\t\t{0: ^24}\n\t\t\t\t\t".format("Predictions")

    for label in lbl:
        pprint_str += "{0: ^7}".format(label)

    pprint_str += "\n"
    try:
        for row_num, row in enumerate(matrix, start=1):
            if row_num == floor((len(matrix) + 1) / 2.0):
                pprint_str += "Actual\t{0: ^10}\t".format(lbl[row_num - 1])
            else:
                pprint_str += "\t\t{0: ^10}\t".format(lbl[row_num - 1])
            for num in row:
                pprint_str += "{0: ^7}".format(num)
            pprint_str += "\n"
    except IndexError:
        pprint_str = "Label array must be the same length as the number of rows/columns in the confusion matrix " \
                     "for it to be printed successfully."

    return pprint_str


for model_path in glob.glob("*_models/*/*/*/*/data/model.h5"):
    model_path_parts = model_path.split("/")
    out_path = "/".join(model_path_parts[:5])
    two_class = "two_class" in model_path
    nn_model = load_model(model_path)

    for data_path in glob.glob("data/{0}/test_data/{1}_*_2000_data.pkl".format(model_path_parts[1], model_path_parts[4])):
        dsn = data_path.split("_")[-2]

        x, y, labels = data_helper.load_data_and_labels(data_path, two_class)
        x_test = numpy.array(x)
        y_test = numpy.array(y)

        scores = nn_model.evaluate(x_test, y_test, verbose=1)

        if two_class:
            with open("{0}/disjoint_test_data_{1}_summary.txt".format(out_path, dsn), mode="w") as summary:
                with open("{0}/disjoint_test_data_{1}_results.json".format(out_path, dsn), mode="w") as results_json:
                    ground_truth = multi_2_single_label(y_test)
                    predictions = multi_2_single_label(cont_2_disc(nn_model.predict(x_test)))
                    c_matrix = confusion_matrix(ground_truth, predictions).tolist()
                    results = {"accuracy": scores[3] * 100, "conf_matrix": c_matrix, "labels": ["CFS", "N-UFS"]}

                    summary.write("Accuracy: {0:.3f}%\n\n".format(scores[3] * 100))
                    summary.write("{0}".format(pprint_c_matrix(c_matrix, ["CFS", "N-UFS"])))
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
        else:
            with open("{0}/disjoint_test_data_{1}_summary.txt".format(out_path, dsn), mode="w") as summary:
                with open("{0}/disjoint_test_data_{1}_results.json".format(out_path, dsn), mode="w") as results_json:
                    ground_truth = multi_2_single_label(y_test)
                    predictions = multi_2_single_label(cont_2_disc(nn_model.predict(x_test)))
                    c_matrix = confusion_matrix(ground_truth, predictions).tolist()
                    results = {"accuracy": scores[3] * 100, "conf_matrix": c_matrix, "labels": ["CFS", "NFS", "UFS"]}

                    summary.write("Accuracy: {0:.3f}%\n\n".format(scores[3] * 100))
                    summary.write("{0}".format(pprint_c_matrix(c_matrix, ["CFS", "NFS", "UFS"])))
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
        print("Finished processing: {0}".format(out_path))

