# this file contains helper functions to deal with the training set and the test set
from itertools import izip
# let's define some global list:
Representation_Of_Indexes_By_Words = {}
Representation_Of_Words_By_Indexes = {}
Representation_Of_classes_By_Indexes = {}
Representation_Of_Indexes_By_classes = {}
# define set let us assign an index to each unique word - avoid giving the same word many indexes
Dictionary_of_classes = set()
Dictionary_of_words = []

import numpy as np

# more to define:
UNK = "UUUNKKK"
WINDOW_START = 'START_WIN'
WINDOW_END = "END_WIN"
NEW_LINE = "\n"
TAB = "\t"


def get_word_embeddings_matrix():
    return np.loadtxt("wordVectors.txt")

def add_class_and_word_to_dics(m_class, word):
    global Dictionary_of_words
    global Dictionary_of_classes
    Dictionary_of_classes.add(m_class)
    Dictionary_of_words.add(word)


def read_train_data(file_name):
    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence_and_tags = []
        for line in content:
            if NEW_LINE == line:
                tagged_sentences.append(sentence_and_tags)
                sentence_and_tags = []
                continue
            line = clean_line(line)
            word, tag = line.split()
            #add_class_and_word_to_dics(tag, word)
            Dictionary_of_classes.add(tag)
            sentence_and_tags.append((word.lower(), tag))
    Dictionary_of_classes.add(UNK)
    #Dictionary_of_words.add(UNK)
    return tagged_sentences


def clean_line(line):
    return line.strip(NEW_LINE).strip().strip(TAB)


def read_dev_data(file_name):
    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence_and_tags = []
        for line in content:
            if NEW_LINE == line:
                tagged_sentences.append(sentence_and_tags)
                sentence_and_tags = []
                continue
            line = clean_line(line)
            word, tag = line.split()
            sentence_and_tags.append((word.lower(), tag))

    return tagged_sentences


def updating_dictionaries_set():
    global Representation_Of_Words_By_Indexes
    global Representation_Of_Indexes_By_Words
    global Representation_Of_classes_By_Indexes
    global Representation_Of_Indexes_By_classes
    global Dictionary_of_words
    print("here2")
    print(len(Dictionary_of_words))
    #Dictionary_of_words.update(set([WINDOW_START, WINDOW_END]))
    print("after2")
    print(len(Dictionary_of_words))
    Representation_Of_Words_By_Indexes = {
        m_word: m_index for m_index, m_word in enumerate(Dictionary_of_words)
    }
    Representation_Of_Indexes_By_Words = {
        m_index: m_word for m_word, m_index in Representation_Of_Words_By_Indexes.iteritems()
    }
    Representation_Of_classes_By_Indexes = {
        m_class: m_index for m_index, m_class in enumerate(Dictionary_of_classes)
    }
    Representation_Of_Indexes_By_classes = {
        m_index: m_class for m_class, m_index in Representation_Of_classes_By_Indexes.iteritems()
    }


def divide_word_class_sequence_into_windows(word_sequences):
    """
       :param word_sequences:
       :return: return an array of windows when each window contain 5 words (as requiredin the assigment)
       """
    windows_array = []
    classes = []
    for sentence in word_sequences:
        words_classes_matrix = []
        # words_classes_matrix = [(WINDOW_START, WINDOW_START), (WINDOW_START, WINDOW_START)]
        words_classes_matrix.extend([(WINDOW_START, WINDOW_START), (WINDOW_START, WINDOW_START)])
        words_classes_matrix.extend(sentence)
        words_classes_matrix.extend([(WINDOW_END, WINDOW_END), (WINDOW_END, WINDOW_END)])
        for i, (cure_word, tag) in enumerate(words_classes_matrix):
            if cure_word != WINDOW_START and cure_word != WINDOW_END:
                new_window = []
                new_window.append(convert_word_to_index(words_classes_matrix[i - 2][0]))
                new_window.append(convert_word_to_index(words_classes_matrix[i - 1][0]))
                new_window.append(convert_word_to_index(cure_word))
                new_window.append(convert_word_to_index(words_classes_matrix[i + 1][0]))
                new_window.append(convert_word_to_index(words_classes_matrix[i - 2][0]))

                windows_array.append(new_window)
                classes.append(Representation_Of_classes_By_Indexes[tag])
    return windows_array, classes


def divide_word_sequence_into_windows(word_sequences):
    """
        for test set which not has the class for each word
        :param word_sequences:
        :return: return an array of windows when each window contain 5 words (as requiredin the assigment)
        """
    windows_array = []
    for sequence in word_sequences:
        # words = [WINDOW_START,WINDOW_START]
        # words.extend(sentence)
        # words.extend([WINDOW_END,WINDOW_END])
        words = []
        # adds start*2, end*2 for each sentence for appropriate windows
        words.append(WINDOW_START)
        words.append(WINDOW_START)
        words.extend(sequence)
        words.append(WINDOW_END)
        words.append(WINDOW_END)
        for index, (cure_word) in enumerate(words):
            if cure_word != WINDOW_START and cure_word != WINDOW_END:
                new_window = []
                new_window.append(convert_word_to_index(words[index - 2]))
                new_window.append(convert_word_to_index(words[index - 1]))
                new_window.append(convert_word_to_index(cure_word))
                new_window.append(convert_word_to_index(words[index + 1]))
                new_window.append(convert_word_to_index(words[index - 2]))

                windows_array.append(new_window)
    return windows_array


def convert_word_to_index(word_to_convert):
    """
       The first step in using an embedding layer is to encode this sentence by indices.
        In this case we assign an index to each unique word
       :param word_to_convert:
       :return: return the index which represent the word.
       """
    if word_to_convert not in Representation_Of_Words_By_Indexes:
        # if the word is not in the list its index is UNK - Unknown
        # words we have not seen in the training action we'll map them to the same vector in
        # the embedding matrix - unk
        return Representation_Of_Words_By_Indexes[UNK]
    else:
        # if the word is in the list, return its index
        return Representation_Of_Words_By_Indexes[word_to_convert]


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

    concat, tags = divide_word_class_sequence_into_windows(tagged_sentences_list)
    return concat, tags


def reading_test_dataset(dataset_to_read):
    # define an empty array to return
    word_sequences = []

    with open(dataset_to_read) as dataset:
        content = dataset.readlines()
        sequence = []
        for cure_line in content:
            if NEW_LINE == cure_line:
                word_sequences.append(sequence)
                # clean buffer
                sequence = []
                # continue yo next iteration of the loop
                continue
            word_in_sequence = cure_line.strip(NEW_LINE).strip()
            sequence.append(word_in_sequence.lower())
    return word_sequences


def get_train_data(file_name):
    global Dictionary_of_words
    global Dictionary_of_classes
    tagged_sentences_list = read_train_data(file_name)
    updating_dictionaries_set()
    windows_array, classes = divide_word_class_sequence_into_windows(tagged_sentences_list)
    return windows_array, classes



def get_word_embeddings_dict_from_file(words_file, vector_file):

    word_embeddings_dict = {}
    for word, vector_line in izip(open(words_file), open(vector_file)):
        word = word.strip(NEW_LINE).strip()
        vector_line = vector_line.strip("\n").strip().split(" ")
        word_embeddings_dict[word] = np.asanyarray(map(float,vector_line))
        Dictionary_of_words.append(word)
    return word_embeddings_dict



def bring_test_data(test_data_set):
    word_sequences = reading_test_dataset(test_data_set)
    return divide_word_sequence_into_windows(word_sequences)

WORD_EMBEDDINGS_DICT = get_word_embeddings_dict_from_file('vocab.txt', 'wordVectors.txt')
E = get_word_embeddings_matrix()