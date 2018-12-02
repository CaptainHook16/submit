import numpy as np
import torch as tc
from itertools import izip


#this file contains helper functions to deal with the training set and the test set

#let's define some global list:
Representation_Of_Indexes_By_Words = {}
Representation_Of_Words_By_Indexes = {}
Representation_Of_classes_By_Indexes ={}
Representation_Of_Indexes_By_classes ={}
#define set let us assign an index to each unique word - avoid giving the same word many indexes
Dictionary_of_classes = set()
Dictionary_of_words = set()


#more to define:
UNK = "UNKNOWN_WORD"
WINDOW_START = "START_WIN"
WINDOW_END = "END_WIN"
NEW_LINE = "\n"
TAB = "\t"

def read_train_data(file_name):

    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence_and_tags = []
        for line in content:
            if line == "\n":
                tagged_sentences.append(sentence_and_tags)
                sentence_and_tags =[]
                continue
            line = line.strip("\n").strip().strip("\t")
            word, tag  = line.split()
            Dictionary_of_classes.add(tag)
            Dictionary_of_words.add(word)
            sentence_and_tags.append((word, tag))
    Dictionary_of_classes.add(UNK)
    Dictionary_of_words.add(UNK)


    return tagged_sentences


def read_dev_data(file_name):
    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence_and_tags = []
        for line in content:
            if line == "\n":
                tagged_sentences.append(sentence_and_tags)
                sentence_and_tags = []
                continue
            line = line.strip("\n").strip().strip("\t")
            word, tag = line.split()

            sentence_and_tags.append((word, tag))

    return tagged_sentences

def read_not_tagged_data(file_name):
    """
    read_not_tagged_data function.
    reads the test file.
    :param file_name: test file name.
    :return: list of sentences.
    """
    sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence = []
        for line in content:
            if line == "\n":
                sentences.append(sentence)
                sentence =[]
                continue
            w = line.strip("\n").strip()
            sentence.append(w)
    return sentences

def load_indexers(word_set, tags_set):
    """
    load_indexers function.
    creates our dicts that helps us to manage the data.
    :param word_set: our words.
    :param tags_set: our tags.
    """
    global  Representation_Of_Words_By_Indexes
    global Representation_Of_Indexes_By_Words
    global Representation_Of_classes_By_Indexes
    global Representation_Of_Indexes_By_classes
    print("here2")
    print(len(Dictionary_of_words))
    word_set.update(set([WINDOW_START, WINDOW_END]))
    print("after2")
    print(len(Dictionary_of_words))
    Representation_Of_Words_By_Indexes = {word : i for i, word in enumerate(word_set)}
    Representation_Of_Indexes_By_Words = {i : word for word, i in Representation_Of_Words_By_Indexes.iteritems()}
    Representation_Of_classes_By_Indexes = {tag : i for i, tag in enumerate(Dictionary_of_classes)}
    Representation_Of_Indexes_By_classes = {i : tag for tag, i in Representation_Of_classes_By_Indexes.iteritems()}



def get_windows_and_tags(tagged_sentences):
    """
    get_windows_and_tags function.
    :param tagged_sentences: examples.
    :return: concat of five window of words and tags.
    """
    concat_words = []
    tags = []
    for sentence in tagged_sentences:
        pad_s = [(WINDOW_START, WINDOW_START), (WINDOW_START, WINDOW_START)]
        pad_s.extend(sentence)
        pad_s.extend([(WINDOW_END, WINDOW_END), (WINDOW_END, WINDOW_END)])
        for i, (word,tag) in enumerate(pad_s):
            if word!=WINDOW_START and word !=WINDOW_END:
                win = get_word_indices_window(pad_s[i - 2][0], pad_s[i - 1][0], word,
                                              pad_s[i + 1][0], pad_s[i + 2][0])
                concat_words.append(win)
                tags.append(Representation_Of_classes_By_Indexes[tag])
    return concat_words, tags

def get_windows(sentences):
    """
    get_windows function.
    :param sentences list.
    :return: concat of five window of words.
    """
    concat_words = []
    for sentence in sentences:
        pad_s = [WINDOW_START,WINDOW_START]
        pad_s.extend(sentence)
        pad_s.extend([WINDOW_END,WINDOW_END])
        for i, (word) in enumerate(pad_s):
            if word != WINDOW_START and word != WINDOW_END:
                win = get_word_indices_window(pad_s[i - 2],pad_s[i - 1],word,pad_s[i + 1],pad_s[i + 2])
                concat_words.append(win)
    return concat_words

def get_word_indices_window(w1,w2,w3,w4,w5):
    """
    get_word_indices_window function.
    :param w1: word1
    :param w2: word2
    :param w3: word3
    :param w4: word4
    :param w5: word5
    :return: concat words window indices.
    """
    win = []
    win.append(get_word_index(w1))
    win.append(get_word_index(w2))
    win.append(get_word_index(w3))
    win.append(get_word_index(w4))
    win.append(get_word_index(w5))
    return win

def get_word_index(word_to_convert):
    """
    get_word_index function.
    :param w: requested word index.
    :return: word index if its in words set, or unk index.
    """
    if word_to_convert in Representation_Of_Words_By_Indexes:
        return Representation_Of_Words_By_Indexes[word_to_convert]
    else:
        return Representation_Of_Words_By_Indexes[UNK]

def get_dev_data(file_name):
    """
    get_tagged_data function.
    :param file_name: file name of the requested data for dev or train.
    :param is_dev:
    :return: data and tags
    """
    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences_list = read_dev_data(file_name)

    concat, tags = get_windows_and_tags(tagged_sentences_list)
    return concat, tags

def get_train_data(file_name,is_dev = False):
    """
    get_tagged_data function.
    :param file_name: file name of the requested data for dev or train.
    :param is_dev:
    :return: data and tags
    """
    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences_list = read_train_data(file_name)
    load_indexers(Dictionary_of_words,Dictionary_of_classes)
    concat, tags = get_windows_and_tags(tagged_sentences_list)
    return concat, tags

# def bring_train_data(train_data):
#     """
#
#     :param train_data:
#     :return:
#     """
#     global Dictionary_of_words
#     global Dictionary_of_classes
#     classes_word_sequences = read_tagged_data(train_data)
#     #add the first words in our dictionary
#     #settings all of the dictionary
#
#
#     load_indexers(Dictionary_of_words, Dictionary_of_classes)
#
#     concat, tags = get_windows_and_tags(tagged_sentences_list)
#     return concat, tags


def get_not_tagged_data(file_name):
    """
    get_not_tagged_data function.
    :param file_name: file name of the requested data for test.
    :return: data
    """
    #global Dictionary_of_words, Dictionary_of_classes
    sentences_list = read_not_tagged_data(file_name)
    concat = get_windows(sentences_list)
    return concat



