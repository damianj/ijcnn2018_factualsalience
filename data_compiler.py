import json
import pickle
import numpy as np
import concurrent.futures
from keras.preprocessing.text import text_to_word_sequence

contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "here's": "here is",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "would",
    "I'd've": "I would have",
    "I'll": " will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "would",
    "i'd've": "i would have",
    "i'll": " will",
    "i'll've": "i will have",
    "i'm": "i am ",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'll": "that will",
    "that've": "that have",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "'em": "them",
    "there're": "there are",
    "there've": "there have",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "there'll": "there will ",
    "they've": "they have",
    "to've": "to have",
    "'til": "until",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    ######################################################################################
    # Data-set specific fixes ############################################################
    ######################################################################################
    "itã\x8fâ‚¬s": "it is",
    "itÃ¢â‚¬â„¢s": "it is",
    "midgetman": "midget man",
    "naãƒâ¯ve": "naive",
    "1990ã\x8fâ‚¬s": "year 1990",
    "30ã\x8fâ‚¬s": "1930",
    "40ã\x8fâ‚¬s": "1940",
    "'40's": "1940",
    "'50's": "1950",
    "'60's": "1960",
    "'87": "1987",
    "'81": "1981",
    "'77": "1977",
    "'83": "1983",
    "'94": "1994",
    "'93": "1993",
    "'97": "1997",
    "'92": "1992",
    "ã¢â€°â¤": "",
    "ã¢â€°â¥mr": "",
    "Ã¢â€°Â¤": "",
    "ã¢â€°â¥who": "who",
    "aayuh": "",
    "mirving": "attack with multiple independently targetable reentry vehicle",
    "kardari": "zardari",
    "countrypeople": "country people",
    "bicta": "",
    "bict": "",
    "l949": "1949",
    "l961": "1961",
    "undefensible": "indefensible",
    "198i": "1981",
    "sholicatchvieli": "shalikashvili",
    "ã¢â‚¬å“we": "we",
    "ã¢â‚¬â\x9d": "",
    "chemomyrdin": "chernomyrdin",
    "chemomyrdin's": "chernomyrdin",
    "revita1ize": "revitalize",
    "arterially": "from the arteries",
    "'80s": "1980",
    "'60s": "1960",
    "hmet": "heavy expanded mobility tactical truck",
    "hmett": "heavy expanded mobility tactical truck",
    "vietnese": "vietnamese",
    "namese": "",
    "''": "",
    "d'amato": "d'amato",
    "shinsheki": "shinseki",
    "exager": "exaggerated",
    "cardash": "radosh",
    "youã¢â‚¬â„¢re": "you are",
    "treasurey": "treasury",
    "itã¢â‚¬â„¢s": "it is",
    "iã¢â‚¬â„¢ll": "i will",
    "ã‚": "",
    "weã¢â‚¬â„¢ll": "we will",
    "ãƒâ¢ã¢â€šâ¬ã¢â‚¬å“": "",
    "270billion": "270 billion",
    "youã¢â‚¬â„¢ve": "you have"
}


def print_progress(iteration, total, prefix='Progress', suffix='Complete', decimals=2, length=100, fill='█'):
    # https://stackoverflow.com/a/34325723/1217580
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print("\r{0} |{1}| {2}% {3}".format(prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print("\n")


def load_embeddings(file, is_pickle=True, is_json=False):
    if (is_pickle and not is_json) or (not is_pickle and is_json):
        if is_pickle:
            with open(file, mode="rb") as embeddings_pkl:
                return pickle.load(embeddings_pkl)
        else:
            with open(file, mode="r") as embeddings_json:
                return json.load(embeddings_json)

    print("Error: Invalid parameters, file must be a pickle or json type file and specified as such in the parameters.")
    return None


def sent_2_embed(sentence, ec):
    def get_embedding_vec(embedding, target):
        try:
            embedding_vec = embedding[target][0]
            return embedding_vec
        except KeyError:
            return embedding_padding_element[0]

    embedding_padding_element = np.zeros((1, 300), np.float32)
    row_padding_element = np.zeros((1, 300 * len(ec)), np.float32)
    tok_sent = text_to_word_sequence(sentence)
    word_list = []

    for word in tok_sent:
        if word in contractions:
            for w in text_to_word_sequence(contractions[word]):
                word_list.append(w)
        elif "'" in word:
            word_list.append(word.split("'")[0])
        else:
            word_list.append(word)

    total_len = len(word_list)

    for word_idx, word in enumerate(word_list):
        word_list[word_idx] = np.concatenate([get_embedding_vec(embed_dict[ec_i], word) for ec_i in ec])

    for add_row in range((200 - len(word_list))):
        word_list.append(row_padding_element[0])

    return total_len, word_list


def data_processor_worker(data_dict=None, ec=None, out_index=None):
    feature_list = ["SENTENCE_SENTIMENT", "ENT_COUNT", "OVERALL_ENT_SENTIMENT", "WIKI_LINK_COUNT", "pos"]
    one_hot = np.zeros((len(feature_list), len(feature_list)), np.float32)
    np.fill_diagonal(one_hot, 1.0)
    pos_ent_dict = dict(zip(feature_list, one_hot))
    verdict_map = {"-1": "Non-factual claim", "0": "Unimportant factual claim", "1": "Important factual claim",
                   -1: "Non-factual claim", 0: "Unimportant factual claim", 1: "Important factual claim"}
    feature_matrix = np.zeros((1, len(feature_list)), np.float32)
    num_words, sent_embed = sent_2_embed(data_dict["sentence"], ec)

    worker_res = {"data": sent_embed, "verdict": verdict_map[data_dict["label"]]}

    feature_matrix += pos_ent_dict["SENTENCE_SENTIMENT"] * data_dict["sentiment"]["score"]
    for ent in data_dict["entities"]:
        feature_matrix += pos_ent_dict["ENT_COUNT"]
        feature_matrix += (pos_ent_dict["OVERALL_ENT_SENTIMENT"] * ent["sentiment"]["score"]) / ent["salience"]
        if ent["wikipedia_url"]:
            feature_matrix += pos_ent_dict["WIKI_LINK_COUNT"]

    for tok in data_dict["tokens"]:
        for pos in tok["part_of_speech"]:
            if tok["part_of_speech"][pos]:
                feature_matrix += pos_ent_dict["pos"] * tok["part_of_speech"][pos] / num_words

    nopad_feature_matrix = feature_matrix[0]
    feature_matrix = feature_matrix[0]

    for extra_row in range((60 * len(ec)) - 1):
        feature_matrix = np.append(feature_matrix, nopad_feature_matrix)

    worker_res["data"] = np.concatenate([worker_res["data"], [feature_matrix]])
    worker_res["data"] = worker_res["data"].tolist()

    return out_index, worker_res


if __name__ == "__main__":
    embed_dict = {1: load_embeddings("embeddings/glove_dict.pkl"),
                  2: load_embeddings("embeddings/google_dict.pkl"),
                  3: load_embeddings("embeddings/facebook_dict.pkl"),
                  4: load_embeddings("embeddings/dep_dict.pkl")}

    embed_combinations = [(1,), (2,), (3,), (4,),
                          (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                          (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
                          (1, 2, 3, 4)]

    embed_comb_name_map = {(1,): "Glove", (2,): "Google", (3,): "Facebook", (4,): "Dep",
                           (1, 2): "GloveGoogle", (1, 3): "GloveFacebook", (1, 4): "GloveDep", (2, 3): "GoogleFacebook",
                           (2, 4): "GoogleDep", (3, 4): "FacebookDep",
                           (1, 2, 3): "GloveGoogleFacebook", (1, 2, 4): "GloveGoogleDep", (1, 3, 4): "GloveFacebookDep",
                           (2, 3, 4): "GoogleFacebookDep",
                           (1, 2, 3, 4): "GloveGoogleFacebookDep"}
    source_data_dict = {"disjoint_2000": "data/disjoint-2000.google-nlp.pkl",
                        "disjoint_500": "data/sentence_data_doped-502.google-nlp.pkl"}

    for d in source_data_dict:
        for embed_comb in embed_combinations:
            final_data = {}
            with open(source_data_dict[d], mode="rb") as pkl_data:
                data = pickle.load(pkl_data)
                data_len = len(data)

                with concurrent.futures.ThreadPoolExecutor(max_workers=28) as executor:
                    sent_processor = {executor.submit(data_processor_worker, data[key], embed_comb, key): key for key in
                                      data}
                    for completed_num, future in enumerate(concurrent.futures.as_completed(sent_processor), start=1):
                        key = sent_processor[future]

                        try:
                            data_index, result = future.result()
                            final_data[data_index] = result
                        except Exception as e:
                            print("{0!r:} generated an exception: {1}".format(key, e))

                        print_progress(completed_num, data_len)

            print("Saving data...")
            with open("data/{0}_embed/test_data/{1}_{2}_data.pkl".format(len(embed_comb),
                                                                         embed_comb_name_map[embed_comb],
                                                                         d), mode="wb") as pkl_out:
                pickle.dump(final_data, pkl_out)
            print("Done!")
