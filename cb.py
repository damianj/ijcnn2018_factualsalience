import pandas as pd
from math import floor
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report
from nltk import word_tokenize
from nltk import pos_tag
from sklearn.externals import joblib
import os
import pickle
import json

vocabulary = pd.read_csv('models/vocabulary_Classification_8000.txt', names=["words"], sep="\t")
count_vect = CountVectorizer(vocabulary=vocabulary.words)
idf_ = pickle.load(open('models/idf_Classification_8000', mode='rb'))
POS = ['MD', 'VBN', 'PRP$', 'CD', 'NNS', 'RBR', 'LS', 'WP', 'JJR', 'RB', 'WP$', 'VBZ', '-LRB-', 'CC', 'JJ', '$', ':',
       'VBG', "''", ',', 'WDT', 'EX', 'PDT', 'RP', '``', 'NNPS', 'NNP', 'FW', 'VB', 'PRP', 'RBS', 'DT', 'WRB', 'NN',
       '.', '-NONE-', 'IN', 'TO', 'UH', 'VBD', 'POS', 'VBP', 'JJS', 'SYM', '(', ')']


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


def getPOSVector(text):#returns a JSON string
    dict_pos_count = {k:0 for k in POS}
    pos_tags = pos_tag(word_tokenize(text))# TextBlob(text).pos_tags#
    for word, tag in pos_tags:
        if tag in dict_pos_count.keys():
            dict_pos_count[tag] += 1
    return dict_pos_count


def getSentiment(text):#getting sentiment from AlchemyAPI is unrealistic
    sentiment = 1.0 #[0(Neg)-2(Pos)]
    return sentiment


def getETVector(text):#getting named entities from AlchemyAPI is unrealistic
    dict_et_count = {"City": 0, "Company": 0, "Continent": 0, "Country": 0, "Crime": 0, "Degree": 0, "Drug": 0, "EntertainmentAward": 0, "Facility": 0, "FieldTerminology": 0, "GeographicFeature": 0, "HealthCondition": 0, "Holiday": 0, "JobTitle": 0, "Movie": 0, "OperatingSystem": 0, "Organization": 0, "Person": 0, "PrintMedia": 0, "Quantity": 0, "Region": 0, "Sport": 0, "SportingEvent": 0, "StateOrCounty": 0, "Technology": 0, "TelevisionShow": 0, "TelevisionStation": 0}
    return dict_et_count


def getCFSScore(transcript="", tokenize_sentence = False, models = [("models/SVM_W_P_Classification_8000.pkl", "W_|P_", 'SVM_8000')]):

    if tokenize_sentence:
        sentences = sent_tokenize(transcript)
    else:
        sentences = [transcript]
    sentences = pd.DataFrame(sentences)
    sentences.columns = ['text']
    sentences['original_text'] = sentences.text
    sentences.text = sentences['original_text'].map( lambda s: s.lower() )
    sentences['sentence_id'] = sentences.index

    sentence_word = count_vect.fit_transform(sentences.text)
    sentence_word = sentence_word.toarray() * idf_
    sentence_word = normalize(sentence_word, axis = 1, norm='l2')

    sentence_word = pd.DataFrame(sentence_word, columns=vocabulary.words)
    sentence_word = pd.concat([sentences.sentence_id, sentence_word], axis = 1)
    sentence_word = sentence_word.set_index('sentence_id')
    sentence_word.columns = sentence_word.columns.map(lambda x: (u'W_' + str(x)))

    sentence_pos = {sentences.sentence_id[i]:getPOSVector(sentences.text[i]) for i in sentences.index}
    sentence_pos = pd.DataFrame(sentence_pos).T
    sentence_pos.columns = sentence_pos.columns.map(lambda x: (u'P_ ' + str(x)))
    sentence_pos.index.names = ['sentence_id']
    sentence_pos['length'] = [len(sentences.ix[id].text.split()) for id in sentences.index]

    sentence_entity_type = {sentences.sentence_id[i]:getETVector(sentences.text[i]) for i in sentences.index}
    sentence_entity_type = pd.DataFrame(sentence_entity_type).T
    sentence_entity_type.columns = sentence_entity_type.columns.map(lambda x: (u'ET_' + str(x)))
    sentence_entity_type.index.names = ['sentence_id']
    sentence_entity_type['sentiment'] = [getSentiment(sentences.text[i]) for i in sentences.index]

    data = pd.concat([sentence_pos, sentence_entity_type], axis = 1)
    data = pd.concat([data, sentence_word], axis = 1)

    for model_path, feature_regex, name in models:
        clf = joblib.load(model_path)
        model_name = clf.__class__.__name__
        data_regex = data.filter(regex=feature_regex)
        data_regex = data_regex.sort_index(axis=1)

        if str(model_name) == 'LinearSVC':
            return clf._predict_proba_lr(data_regex)[0]
        else:
            return clf._predict_proba_lr(data_regex)[0]


with open("data/sentence_data_doped-502.google-nlp.pkl", mode="rb") as data_pkl:
    data = pickle.load(data_pkl)

ground_truth = []
predictions = []
test_sents = ["I ate apples.",
"I ate 2 apples.",
"I ate 500 apples.",
"Iraq does not have weapons of mass destruction.",
"Millions of illegal immigrants voted last year.",
"The 534 apples spread out across 3 tables had been left out for 1 day 9 hours and 23 minutes.",
"The U.S. allowed 320 million illegal immigrants to vote in the 2016 elections.",
"important 6 615651 956528 967151 millions billions very 7673 88356 low 234 65 766 234 789 9000 3452."]
for d in test_sents:
    print(d, getCFSScore(d))
    # ground_truth.append(int(data[d]["label"]))
    # predictions.append(int(getCFSScore(data[d]["sentence"])))
exit(0)
# Only for the 500 nonsensical sentences dataset, it needs it or there is a calculation error in precision and recall
# due to no labels being associated to the CFS or UFS classes
# ground_truth.append(1)
# predictions.append(int(getCFSScore("One million illegal immigrants voted in the 2017 elections.")))
# ground_truth.append(0)
# predictions.append(int(getCFSScore("Today I ate green eggs and ham.")))

with open("data/claimspotter/summary_10000_500.txt", mode="w") as summary:
    with open("data/claimspotter/results_10000_500.json", mode="w") as results_json:
        c_matrix = confusion_matrix(ground_truth, predictions).tolist()
        acc = accuracy_score(ground_truth, predictions) * 100
        results = {"accuracy": acc, "conf_matrix": c_matrix, "labels": ["NFS", "UFS", "CFS"]}

        summary.write("Accuracy: {0:.3f}%\n\n".format(acc))
        summary.write("{0}".format(pprint_c_matrix(c_matrix, ["NFS", "UFS", "CFS"])))
        summary.write("\n\nMeasure\t\t\tCFS\t\t\tNFS\t\t\tUFS\t\tWeighted Average\n")

        nfs, ufs, cfs = precision_score(ground_truth, predictions, average=None)
        w_a = precision_score(ground_truth, predictions, average="weighted")
        results["precision"] = {"csf": cfs, "nfs": nfs, "ufs": ufs, "weighted_average": w_a}
        summary.write("Precision\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\t\t{3:.5f}\n".format(cfs, nfs, ufs, w_a))

        nfs, ufs, cfs = recall_score(ground_truth, predictions, average=None)
        w_a = recall_score(ground_truth, predictions, average="weighted")
        results["recall"] = {"csf": cfs, "nfs": nfs, "ufs": ufs, "weighted_average": w_a}
        summary.write("Recall\t\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\t\t{3:.5f}\n".format(cfs, nfs, ufs, w_a))

        nfs, ufs, cfs = f1_score(ground_truth, predictions, average=None)
        w_a = f1_score(ground_truth, predictions, average="weighted")
        results["f1_score"] = {"csf": cfs, "nfs": nfs, "ufs": ufs, "weighted_average": w_a}
        summary.write("F1-Score\t{0:.5f}\t\t{1:.5f}\t\t{2:.5f}\t\t{3:.5f}\n".format(cfs, nfs, ufs, w_a))
        json.dump(results, results_json)
