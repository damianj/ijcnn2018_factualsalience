import pickle
import concurrent.futures
import numpy as np
import parse_tree_data_compiler


def print_progress(iteration, total, prefix='Progress', suffix='Complete', decimals=2, length=100, fill='█'):
    # https://stackoverflow.com/a/34325723/1217580
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print("\r{0} |{1}| {2}% {3}".format(prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


def load_data_worker(data, to_two_class=False):
    if data["verdict"]:
        if to_two_class and data["verdict"] != "Important factual claim":
            return data["data"], "Irrelevant claim"
        else:
            return data["data"], data["verdict"]


def load_data_and_labels(filename, to_two_class=False, on_the_fly=False):
    x_raw = []
    labels = []
    
    print("Opening data file (might take a while depending on size)...")
    with open(filename, mode="rb") as pkl_file:
        data = pickle.load(pkl_file)
    data_len = len(data)

    print("Processing data...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=28) as executor:
        data_processor = {executor.submit(load_data_worker, data[key], to_two_class): key for key in data}
        for num_completed, future in enumerate(concurrent.futures.as_completed(data_processor), start=1):
            key = data_processor[future]
            try:
                x_data, verdict_text = future.result()
                x_raw.append(x_data)
                labels.append(verdict_text)
            except Exception as e:
                print("{0!r:} generated an exception: {1}".format(key, e))

            print_progress(num_completed, data_len)

    x_raw = np.array(x_raw)

    labels_list = sorted(list(set(labels)))
    one_hot = np.zeros((len(labels_list), len(labels_list)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels_list, one_hot))
    y_raw = [label_dict[label] for label in labels]

    return x_raw, np.array(y_raw), labels_list
