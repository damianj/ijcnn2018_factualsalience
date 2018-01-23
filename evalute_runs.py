import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import glob
import json



def scatter3d(x, y, z, size, cs, colorsMap='tab20b'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=size, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap,label='Test')
    plt.show()


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0.0


highest_acc = [None, None, 0]
prefix = "*_models/"
result_sets = ["2000", "2000"]
mappings = {"classifier_models": 0, "regression_models": 1,
            "1_embed": 1, "2_embed": 2, "3_embed": 3, "4_embed": 4,
            "regular_data": 50, "enhanced_data": 200,
            "two_class": 1, "three_class": 2}

for r_set in result_sets:
    precision = {"enhanced": {"two_class": {"nfs": [], "csf": [], "weighted_average": []}, "three_class": {"nfs": [], "ufs": [], "csf": [], "weighted_average": []}}, "regular": {"two_class": {"nfs": [], "csf": [], "weighted_average": []}, "three_class": {"nfs": [], "ufs": [], "csf": [], "weighted_average": []}}}
    recall = {"enhanced": {"two_class": {"nfs": [], "csf": [], "weighted_average": []}, "three_class": {"nfs": [], "ufs": [], "csf": [], "weighted_average": []}}, "regular": {"two_class": {"nfs": [], "csf": [], "weighted_average": []}, "three_class": {"nfs": [], "ufs": [], "csf": [], "weighted_average": []}}}
    f1 = {"enhanced": {"two_class": {"nfs": [], "csf": [], "weighted_average": []}, "three_class": {"nfs": [], "ufs": [], "csf": [], "weighted_average": []}}, "regular": {"two_class": {"nfs": [], "csf": [], "weighted_average": []}, "three_class": {"nfs": [], "ufs": [], "csf": [], "weighted_average": []}}}

    results_ordered = []
    x =[]
    y = []
    z = []
    sz = []
    color = []
    for results_path in glob.glob("{0}/*/*/*/*/disjoint_test_data_{1}_results.json".format(prefix, r_set)):
        p = results_path.split("/")

        with open(results_path, mode="r") as results_json:
            results = json.load(results_json)
            results_ordered.append((results["accuracy"], "{0}_{1}_{2}_{3}_{4}".format(p[0], p[1], p[2], p[3], p[4])))
            x.append(mappings[p[0]])
            y.append(results["accuracy"])
            z.append(mappings[p[1]])
            sz.append(mappings[p[2]])
            color.append(mappings[p[3]])
            if "enhanced" in results_path:
                if "two_class" in results_path:
                    for item in results["precision"]:
                            precision["enhanced"]["two_class"][item].append(results["precision"][item])
                            recall["enhanced"]["two_class"][item].append(results["recall"][item])
                            f1["enhanced"]["two_class"][item].append(results["f1_score"][item])
                else:
                    for item in results["precision"]:
                            precision["enhanced"]["three_class"][item].append(results["precision"][item])
                            recall["enhanced"]["three_class"][item].append(results["recall"][item])
                            f1["enhanced"]["three_class"][item].append(results["f1_score"][item])
            else:
                if "two_class" in results_path:
                    for item in results["precision"]:
                            precision["regular"]["two_class"][item].append(results["precision"][item])
                            recall["regular"]["two_class"][item].append(results["recall"][item])
                            f1["regular"]["two_class"][item].append(results["f1_score"][item])
                else:
                    for item in results["precision"]:
                            precision["regular"]["three_class"][item].append(results["precision"][item])
                            recall["regular"]["three_class"][item].append(results["recall"][item])
                            f1["regular"]["three_class"][item].append(results["f1_score"][item])

    results_ordered.sort(key=lambda t: t[0], reverse=True)
    for r in results_ordered:
        print(r)

    exit()
    for c in precision:
        for k in precision[c]:
            print(c, k)
            for j in precision[c][k]:
                print("precision", j, safe_divide(sum(precision[c][k][j]), len(precision[c][k][j])))
                print("recall", j, safe_divide(sum(recall[c][k][j]), len(recall[c][k][j])))
                print("f1", j, safe_divide(sum(f1[c][k][j]), len(f1[c][k][j])))
            print("\n")

    regression_models_e2 = []
    regression_models_e3 = []
    regression_models_r2 = []
    regression_models_r3 = []
    classifier_models_e2 = []
    classifier_models_e3 = []
    classifier_models_r2 = []
    classifier_models_r3 = []

    for r in results_ordered:
        if "regression" in r[1]:
            if "enhanced" in r[1]:
                if "two_class" in r[1]:
                    regression_models_e2.append((r[0], r[1].split("_")[2]))
                else:
                    regression_models_e3.append((r[0], r[1].split("_")[2]))
            else:
                if "two_class" in r[1]:
                    regression_models_r2.append((r[0], r[1].split("_")[2]))
                else:
                    regression_models_r3.append((r[0], r[1].split("_")[2]))
                
        elif "classifier" in r[1]:
            if "enhanced" in r[1]:
                if "two_class" in r[1]:
                    classifier_models_e2.append((r[0], r[1].split("_")[2]))
                else:
                    classifier_models_e3.append((r[0], r[1].split("_")[2]))
            else:
                if "two_class" in r[1]:
                    classifier_models_r2.append((r[0], r[1].split("_")[2]))
                else:
                    classifier_models_r3.append((r[0], r[1].split("_")[2]))

    regression_models_e2.sort(key=lambda t: t[0], reverse=False)
    regression_models_e3.sort(key=lambda t: t[0], reverse=False)
    regression_models_r2.sort(key=lambda t: t[0], reverse=False)
    regression_models_r3.sort(key=lambda t: t[0], reverse=False)
    classifier_models_e2.sort(key=lambda t: t[0], reverse=False)
    classifier_models_e3.sort(key=lambda t: t[0], reverse=False)
    classifier_models_r2.sort(key=lambda t: t[0], reverse=False)
    classifier_models_r3.sort(key=lambda t: t[0], reverse=False)

    with open("disjoint_{0}_regression_e2.csv".format(r_set), mode="w") as txt_out:
        for r in regression_models_e2:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_regression_e3.csv".format(r_set), mode="w") as txt_out:
        for r in regression_models_e3:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_regression_r2.csv".format(r_set), mode="w") as txt_out:
        for r in regression_models_r2:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_regression_r3.csv".format(r_set), mode="w") as txt_out:
        for r in regression_models_r3:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_classifier_e2.csv".format(r_set), mode="w") as txt_out:
        for r in classifier_models_e2:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_classifier_e3.csv".format(r_set), mode="w") as txt_out:
        for r in classifier_models_e3:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_classifier_r2.csv".format(r_set), mode="w") as txt_out:
        for r in classifier_models_r2:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    with open("disjoint_{0}_classifier_r3.csv".format(r_set), mode="w") as txt_out:
        for r in classifier_models_r3:
            txt_out.write("{0},{1}\n".format(r[1], r[0]))

    scatter3d(x, y, z, sz, color)
