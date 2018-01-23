from keras.models import load_model
import pickle
import numpy as np
import json
from keras.preprocessing.text import text_to_word_sequence
from google.cloud import language

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

trinary_mappings = {0: "Important factual claim", 1: "Non-factual claim", 2: "Unimportant factual claim"}
binary_mappings = {0: "Important factual claim", 1: "Irrelevant claim"}
feature_list = ["SENTENCE_SENTIMENT", "ENT_COUNT", "OVERALL_ENT_SENTIMENT", "WIKI_LINK_COUNT", "pos"]
one_hot = np.zeros((len(feature_list), len(feature_list)), np.float32)
np.fill_diagonal(one_hot, 1.0)
pos_ent_dict = dict(zip(feature_list, one_hot))
padding_element = np.zeros((1, 300), np.float32)


def check_for_key(d, k):
    if k in d:
        return d[k]
    return None


def get_graph_url(graph_id):
    if graph_id:
        return "https://www.google.com/search?q=knowledge+graph+search+api&kponly&kgmid={0}".format(graph_id)
    return None


def analyze(sentence):
    # https://goo.gl/E17iJ7
    result = None
    features = {"extract_syntax": True,
                "extract_entities": True,
                "extract_document_sentiment": True,
                "extract_entity_sentiment": True,
                "classify_text": False}
    try:
        language_client = language.LanguageServiceClient()
        document = language.types.language_service_pb2.Document(content=sentence,
                                                                type=language.enums.Document.Type.PLAIN_TEXT)
        annotations = language_client.annotate_text(document=document, features=features)
        entity_list = []

        for entity in annotations.entities._values:
            entity_list.append({"name": entity.name,
                                "type": entity.Type.DESCRIPTOR.values[entity.type].name,
                                "knowledge_graph": {"id": check_for_key(entity.metadata, "mid"),
                                                    "url": get_graph_url(check_for_key(entity.metadata, "mid"))},
                                "wikipedia_url": check_for_key(entity.metadata, "wikipedia_url"),
                                "salience": entity.salience,
                                "mentions": [{"text": {"content": m.text.content,
                                                       "offset": m.text.begin_offset},
                                              "type": m.Type.DESCRIPTOR.values[m.type].name,
                                              "sentiment": {"magnitude": m.sentiment.magnitude,
                                                            "score": m.sentiment.score}
                                              } for m in entity.mentions],
                                "sentiment": {"magnitude": entity.sentiment.magnitude,
                                              "score": entity.sentiment.score}})

        result = {"sentiment": {"score": annotations.document_sentiment.score,
                                "magnitude": annotations.document_sentiment.magnitude},
                  "entities": entity_list,
                  "tokens": [{"edge_index": token.dependency_edge.head_token_index,
                              "edge_label": token.dependency_edge.label,
                              "lemma": token.lemma,
                              "part_of_speech": {"aspect": token.part_of_speech.aspect,
                                                 "case": token.part_of_speech.case,
                                                 "form": token.part_of_speech.form,
                                                 "gender": token.part_of_speech.gender,
                                                 "mood": token.part_of_speech.mood,
                                                 "number": token.part_of_speech.number,
                                                 "person": token.part_of_speech.person,
                                                 "proper": token.part_of_speech.proper,
                                                 "reciprocity": token.part_of_speech.reciprocity,
                                                 "tag": token.part_of_speech.tag,
                                                 "tense": token.part_of_speech.tense,
                                                 "voice": token.part_of_speech.voice},
                              "text_content": token.text.content} for token in annotations.tokens._values]}

        return result
    except Exception as e:
        print(e)
        exit(0)


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


def sent_2_embed(sentence, g_feat):
    def get_embedding_vec(embedding, target):
        try:
            embedding_vec = embedding[target][0]
            return embedding_vec
        except KeyError:
            return embedding_padding_element[0]

    embedding_padding_element = np.zeros((1, 300), np.float32)
    row_padding_element = np.zeros((1, 300 * len(embeddings)), np.float32)
    feature_matrix = np.zeros((1, len(feature_list)), np.float32)
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

    num_words = len(word_list)

    for word_idx, word in enumerate(word_list):
        word_list[word_idx] = np.concatenate([get_embedding_vec(e, word) for e in embeddings])

    for add_row in range((200 - len(word_list))):
        word_list.append(row_padding_element[0])

    feature_matrix += pos_ent_dict["SENTENCE_SENTIMENT"] * g_feat["sentiment"]["score"]
    for ent in g_feat["entities"]:
        feature_matrix += pos_ent_dict["ENT_COUNT"]
        feature_matrix += (pos_ent_dict["OVERALL_ENT_SENTIMENT"] * ent["sentiment"]["score"]) / ent["salience"]
        if ent["wikipedia_url"]:
            feature_matrix += pos_ent_dict["WIKI_LINK_COUNT"]

    for tok in g_feat["tokens"]:
        for pos in tok["part_of_speech"]:
            if tok["part_of_speech"][pos]:
                feature_matrix += pos_ent_dict["pos"] * tok["part_of_speech"][pos] / num_words

    nopad_feature_matrix = feature_matrix[0]
    feature_matrix = feature_matrix[0]
    for x in range((60 * len(embeddings)) - 1):
        feature_matrix = np.append(feature_matrix, nopad_feature_matrix)

    word_list = np.concatenate([word_list, [feature_matrix]])
    word_list = word_list.tolist()
    return word_list


def model_tester(test_text=None, google_features=None, m=None):
    test_text = np.array([sent_2_embed(test_text, google_features), ])
    reconstructed_model = m
    prediction_list = reconstructed_model.predict(x=test_text)
    num_classes = len(prediction_list[0])

    if num_classes == 2:
        mappings = binary_mappings
    else:
        mappings = trinary_mappings

    for i, pred in enumerate(prediction_list[0]):
        print(mappings[i], ":", pred)

    print(prediction_list)


glove_embeddings = load_embeddings("embeddings/dep_dict.pkl")
google_embeddings = load_embeddings("embeddings/google_dict.pkl")
facebook_embeddings = load_embeddings("embeddings/facebook_dict.pkl")
dep_embeddings = load_embeddings("embeddings/dep_dict.pkl")
embeddings = []

model_path = "regression_models/3_embed/enhanced_data/two_class/GloveGoogleDep/data/model.h5"

if "Glove" in model_path:
    embeddings.append(glove_embeddings)
if "Google" in model_path:
    embeddings.append(google_embeddings)
if "Facebook" in model_path:
    embeddings.append(facebook_embeddings)
if "Dep" in model_path:
    embeddings.append(dep_embeddings)


test_texts = ["important 6 615651 956528 967151 millions billions very 7673 88356 low.",
              "Today I ate eggs and it was 59 degrees outside.",
              "The U.S. allowed 320 million illegal immigrants to vote in the 2016 elections.",
              "We are serving a million more families now.",
              "Today in the U.S. there are 20 million starving children, and 3 die from starvation each day.",
              "In my opinion there wasn't any interference from Russia in the election.",
              "There are not any terrorists living in the United States.",
              "More than 5 Americans are HIV-positive.",
              "Over a million and a quarter Americans are HIV-positive."]

test_text = "500 important 6 615651 956528 967151 millions billions very 7673 88356 low."

g_features = analyze(test_text)
current_model = load_model(model_path)
model_tester(test_text, g_features, current_model)

# current_model = load_model(model_path)
# visualize(current_model)
