import random


def generate_number_word(include_words=True):
    number_word = ""
    dummy_chars = ["i", "am", "many", "very", "low", "high", "billion", "you", "important", "no", "yes", "million"]

    if include_words and random.random() > 0.80:
        number_word += "{0}".format(dummy_chars[random.randint(0, len(dummy_chars) - 1)])
    else:
        for word_len in range(random.randint(1, 15)):
            number_word += "{0}".format(random.randint(0, 9))

    return number_word


def generate_word():
    letter_word = ""
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z", "ñ"]

    for word_len in range(random.randint(1, 20)):
        letter_word += "{0}".format(alphabet[random.randint(0, len(alphabet) - 1)])

    return letter_word


def generate_char_sequence():
    letter_word = ""
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z", "ñ"]

    for word_len in range(random.randint(1, 170)):
        letter_word += "{0} ".format(alphabet[random.randint(0, len(alphabet) - 1)])

    return letter_word.strip()


def generate_number_sentence():
    number_sentence = ""

    for sentence_len in range(random.randint(1, 170)):
        number_sentence += "{0} ".format(generate_number_word())

    return number_sentence.strip()

