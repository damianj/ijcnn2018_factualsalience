import json
import pickle
import time
from nonsense_gen import generate_number_sentence
#######################################################################################
# Need to install google-cloud-sdk                                                    #
# Archlinux:                                                                          #
#   Install this AUR package https://aur.archlinux.org/packages/google-cloud-sdk/     #
#   pacaur -S google-cloud-sdk                                                        #
#   Follow rest of instructions at https://cloud.google.com/sdk/docs/quickstart-linux #
# Debian/Ubuntu:                                                                      #
#   Follow instructions at https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu #
#######################################################################################

# pip install google-cloud-language
from google.cloud import language


def check_for_key(d, k):
    if k in d:
        return d[k]
    return None


def get_graph_url(graph_id):
    if graph_id:
        return "https://www.google.com/search?q=knowledge+graph+search+api&kponly&kgmid={0}".format(graph_id)
    return None


def analyze(data):
    # https://goo.gl/E17iJ7
    features = {"extract_syntax": True,
                "extract_entities": True,
                "extract_document_sentiment": True,
                "extract_entity_sentiment": True,
                "classify_text": False}

    while True:
        try:
            language_client = language.LanguageServiceClient()
            document = language.types.language_service_pb2.Document(content=data["text"],
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

            return {"sentence": data["text"],
                    "label": data["label"],
                    "sentiment": {"score": annotations.document_sentiment.score,
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
        except Exception as e:
            print(e)
            time.sleep(180)


def process_file(f_path, out_name, is_pickle=False):
    results = {}

    if is_pickle:
        with open(f_path, mode="rb") as data_file:
            data = pickle.load(data_file)
    else:
        with open(f_path, mode="r") as data_file:
            data = json.load(data_file)

    for i, item in enumerate(data):
        time.sleep(1)
        print("Processing sentence {0}.\n".format(i))
        results[i] = analyze({"text": item["text"], "label": item["label"]})

    with open("data/{0}.google-nlp.pkl".format(out_name), mode="wb") as pkl_out:
        pickle.dump(results, pkl_out)


def process_sentences(sentence_list, out_name):
    results = {}

    for i, (sent, label) in enumerate(sentence_list):
        time.sleep(1)
        print("Processing sentence {0}.\n".format(i))
        results[i] = analyze({"text": sent, "label": label})

    with open("data/{0}.google-nlp.pkl".format(out_name), mode="wb") as pkl_out:
        pickle.dump(results, pkl_out)


def concatenate_results(file_1, file_2, out_name, is_pickle=False):
    results = {}

    if is_pickle:
        with open(file_1, mode="rb") as data_file:
            data_1 = pickle.load(data_file)

        with open(file_2, mode="rb") as data_file:
            data_2 = pickle.load(data_file)
    else:
        with open(file_1, mode="r") as data_file:
            data_1 = json.load(data_file)

        with open(file_2, mode="r") as data_file:
            data_2 = json.load(data_file)

    for key in data_1:
        results[int(key)] = data_1[key]

    for index, key in enumerate(data_2, start=len(data_1)):
        results[index] = data_2[key]

    with open("data/{0}.google-nlp.pkl".format(out_name), mode="wb") as pkl_out:
        pickle.dump(results, pkl_out)


# sentences = [(generate_number_sentence(), "-1") for num_sent in range(500)]
#process_sentences([("Today I ate green eggs and ham.", 0),
                   #("One million illegal immigrants voted in the 2017 elections.", 1)], "extra_sentences")

# process_file("data/disjoint_2000.pkl", "disjoint-2000", True)

concatenate_results("data/extra_sentences.google-nlp.pkl", "data/sentence_data_doped-500.google-nlp.pkl", "entence_data_doped-502", True)
