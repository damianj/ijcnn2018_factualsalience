import pickle
from itertools import product

import nltk
import numpy as np
import pandas as pd
import snowballstemmer
import sklearn.metrics as mt
from nltk import pos_tag
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

PCA_components = 500
POS = ['MD', 'VBN', 'PRP$', 'CD', 'NNS', 'RBR', 'LS', 'WP', 'JJR', 'RB', 'WP$', 'VBZ', '-LRB-', 'CC', 'JJ', '$', ':',
       'VBG', "''", ',', 'WDT', 'EX', 'PDT', 'RP', '``', 'NNPS', 'NNP', 'FW', 'VB', 'PRP', 'RBS', 'DT', 'WRB', 'NN',
       '.', '-NONE-', 'IN', 'TO', 'UH', 'VBD', 'POS', 'VBP', 'JJS', 'SYM', '(', ')']

file_name = 'data/Classification_10000.json'


def getNgram(d, n):
    tfidf_vect = TfidfVectorizer(min_df=1, ngram_range=(n,n))
    sentence_term_tfidf = tfidf_vect.fit_transform(d.text)
    sentence_term_df = pd.DataFrame(sentence_term_tfidf.todense(), columns=np.sort(list(tfidf_vect.vocabulary_.keys())))
    pd.DataFrame(sentence_term_df.columns).to_csv("models/vocabulary_" + file_name.split("/")[-1] + ".txt", encoding='utf-8',
                                                  index=False, header=False, index_label=False, sep="\t")
    sentence_term_df.columns = sentence_term_df.columns.map(lambda x: 'W_' + str(n) + '_' + x)
    pickle.dump(tfidf_vect.idf_, open("models/idf_" + file_name.split("/")[-1], "wb"))  # storing the vocabulary
    return sentence_term_df


def getPOS(d, n):
    POSn = {x:0 for x in product(POS, repeat=n)}
    data_pos = pd.DataFrame(columns=POSn.keys())
    data_pos.columns = data_pos.columns.map(lambda x: 'P_'+str(n)+'_'+'_'.join(x) )
    for idx,text in enumerate(d.text):
        POSn = {x:0 for x in product(POS, repeat=n)}
        token_pos = pos_tag(nltk.word_tokenize(text))
        token_pos = [y for (x,y) in token_pos]
        token_pos = zip(*[token_pos[i:] for i in range(n)])
        for tp in token_pos:
            POSn[tp] += 1
        data_pos.loc[idx] = list(POSn.values())
    return data_pos


def getET():
    raw_sentence_entity_type = pd.read_csv('raw_Sentence_Entity_Type.csv')
    sentence_entity_type = pd.DataFrame(
        {k: list(g.drop('sentence_id', axis=1).set_index('entity_type').to_dict().values())[0] for k, g in
         raw_sentence_entity_type.groupby("sentence_id")}).T
    sentence_entity_type.columns = sentence_entity_type.columns.map(lambda x: 'ET_' + str(x))
    sentence_entity_type['sentence_id'] = sentence_entity_type.index
    return sentence_entity_type


def getSentimentLength():
    sentence_sentiment_length = pd.read_csv('raw_Sentence_Sentiment_Length.csv')
    sentence_sentiment_length.columns = sentence_sentiment_length.columns.map(lambda x: 'A_' + str(x))
    sentence_sentiment_length.rename(columns={'A_sentence_id': 'sentence_id'}, inplace=True)
    return sentence_sentiment_length


def getPCA(d, n_components=100):
    pca = PCA(n_components)
    data_pca = pca.fit_transform(d)
    data_pca = pd.DataFrame(data_pca)
    data_pca.columns = data_pca.columns.map(lambda x: 'PCA' + '_' + str(x))
    return data_pca


def trainModels(d, clfs, features, labels):
    models = []
    for feature_regex, feature_types in features:
        for clf, name in clfs:
            clf.fit(d.filter(regex=feature_regex), d[labels])
            joblib.dump(clf, 'models/' + name + '_' + feature_types + '_' + file_name.split("/")[-1] + '.pkl')
            models += [(name, (feature_regex, feature_types), clf)]
            print(name + '_' + feature_types + " complete")
    return models


def evaluate(data, clfs, features, labels, fold):
    a = list(data[labels])
    b = [0] * np.unique(data[labels])
    mask = [0] * len(a)
    for i in range(len(a)):
        if a[i] == -1:
            mask[i] = b[0]
            b[0] = (b[0] + 1) % fold
        elif a[i] == 0:
            mask[i] = b[1]
            b[1] = (b[1] + 1) % fold
        elif a[i] == 1:
            mask[i] = b[2]
            b[2] = (b[2] + 1) % fold
    data.loc[:, 'msk'] = pd.Series(mask, index=data.index)

    results = pd.DataFrame(columns=(
    'n_sample', 'algorithm', 'features', 'p_NFS', 'p_UFS', 'p_CFS', 'p_wavg', 'r_NFS', 'r_UFS', 'r_CFS', 'r_wavg',
    'f_NFS', 'f_UFS', 'f_CFS', 'f_wavg'))

    for feature_regex, feature_types in features:
        for clf, name in clfs:
            print('##########' + name + '###' + feature_types + '#######')
            precision, recall, f1, kappa_value, prf_w, cm = np.zeros((1, len(np.unique(data[labels])))), np.zeros(
                (1, len(np.unique(data[labels])))), np.zeros((1, len(np.unique(data[labels])))), 0, np.zeros(
                (1, 3)), np.zeros((3, 3))  # , roc_auc = np.zeros((1,len(np.unique(data_regex.verdict))))
            for i in range(fold):
                train = data.loc[data.msk != i]
                test = data.loc[data.msk == i]
                clf.fit(train.filter(regex=feature_regex), train[labels])

                prediction = clf.predict(test.filter(regex=feature_regex))
                print('##########Classification Report##########')
                print(mt.classification_report(test[labels], prediction))
                print('##########Confusion Matrix##########')
                cm += mt.confusion_matrix(test[labels], prediction)
                print(mt.confusion_matrix(test[labels], prediction))
                print('')

                precision += mt.precision_score(test[labels], prediction, average=None)
                recall += mt.recall_score(test[labels], prediction, average=None)
                f1 += mt.f1_score(test[labels], prediction, average=None)
                prf_w += mt.precision_recall_fscore_support(test[labels], prediction, average='weighted')[0:3]

            precision /= fold
            recall /= fold
            f1 /= fold
            prf_w /= fold
            print('##########Overall Confusion Matrix##########')
            print(cm)
            results.loc[len(results)] = [data.shape[0], name, feature_types, precision[0][0], precision[0][1],
                                         precision[0][2], prf_w[0][0], recall[0][0], recall[0][1], recall[0][2],
                                         prf_w[0][1], f1[0][0], f1[0][1], f1[0][2], prf_w[0][2]]  # , roc_auc[0][ind]
    return results


data = pd.read_json(file_name, encoding='utf-8')
print(data.shape)

data = pd.concat([data, getNgram(data, 1)], axis=1)
print(data.shape)

data = pd.concat([data, getPOS(data, 1)], axis=1)
print(data.shape)

# data = pd.merge(data, getET(), how='left', on=['sentence_id'])
# data.fillna(0, inplace=True)
# print(data.shape)

# data = pd.merge(data, getSentimentLength(), how='left', on=['sentence_id'])
# data.fillna(0, inplace=True)
# print(data.shape)

# data = pd.concat([data, getPCA(data.filter(regex='A_|W_1|P_1|ET_'), PCA_components)], axis=1)
# print(data.shape)

# columns = data.filter(regex='A_|W_1|P_1|ET_|PCA_').columns
# data[columns] = MinMaxScaler().fit_transform(data[columns])
# print(data.shape)


joblib.dump(data, './models/DF_' + file_name.split("/")[-1] + '.pkl')
clfs = [(LinearSVC(), "SVM")]
features = [('W_1|P_1', 'W_P')]
results = pd.DataFrame()

for n_sample in [10231, ]:
    results = pd.concat([results, evaluate(data.sample(n_sample), clfs, features, 'label', 4)], axis=0)

results.to_csv('results_' + file_name.split("/")[-1] + '.csv', index=False)

models = trainModels(data, clfs, [('W_1|P_1', 'W_P')], 'label')
