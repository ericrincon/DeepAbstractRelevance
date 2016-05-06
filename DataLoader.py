import numpy as np
import pandas as pd
import nltk

import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def get_all_files(path):
    file_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:

            # Make sure hidden files do not make into the list
            if name[0] == '.':
                continue
            file_paths.append(os.path.join(path, name))
    return file_paths


def df2vocab(data_frames, use_lowercase=False):
    if type(data_frames) is not list:
        data_frames = [data_frames]

    cv = CountVectorizer(lowercase=use_lowercase)
    text = []

    for data_frame in data_frames:
        text.extend(data_frame.values.tolist())
    cv.fit(text)

    return cv.vocabulary_


def text2seq(texts, vocab, sizes):
    if type(texts) is not list:
        texts = [texts]

    tokenized_texts = []
    seqs = []
    vocab_size = len(vocab)

    # Add parameter PADDING to vocab????
    vocab['PADDING'] = vocab_size
    vocab_size += 1

    for text in texts:
        tokenized_texts.append(nltk.tokenize.word_tokenize(text))

    for i, tokenized_text in enumerate(tokenized_texts):
        max_size = sizes[i]

        seq = []

        for token in tokenized_text:
            if token in vocab:
                seq.append(vocab[token])
            else:
                vocab[token] = vocab_size
                seq.append(vocab_size)
                vocab_size += 1

        n = len(seq)

        if not n == max_size:
            padding_size = max_size - n

            padding_vector = [vocab['PADDING'] for j in range(padding_size)]

            seq = seq + padding_vector

        seqs.append(np.array(seq))

    return seqs


def get_data_as_seq(w2v, w2v_vector_len, max_words):
    file_paths = get_all_files('Data')
    sizes = [max_words['text'], max_words['mesh'], max_words['title']]
    X_list, label_list = [], []
    embeddings = []
    n_examples = 0

    for file_path in file_paths:
        data_frame = pd.read_csv(file_path)

        abstract_text, abstract_labels = extract_abstract_and_labels(data_frame)
        mesh_terms, title = extract_mesh_and_title(data_frame)
        vocab_dict = df2vocab([abstract_text, mesh_terms, title])

        text_seq_examples = []
        title_seq_examples = []
        mesh_seq_examples = []

        X, y = [], []
        labels = []

        for i in range(abstract_text.shape[0]):
            abstract = abstract_text.iloc[i]

            if abstract == 'MISSING':
                continue
            else:
                mesh = mesh_terms[i]
                abstract_title = title[i]
                labels.append(abstract_labels.iloc[i])

            text_seq, mesh_seq, title_seq = text2seq([abstract, mesh, abstract_title], vocab_dict, sizes)

            text_seq_examples.append(np.array(text_seq))
            title_seq_examples.append(np.array(title_seq))
            mesh_seq_examples.append(np.array(mesh_seq))

            y.append(labels)

        embedding = build_embeddings(vocab_dict, w2v, w2v_vector_len)
        embeddings.append(embedding)
        X_list.append([np.array(text_seq_examples), np.array(title_seq_examples), np.array(mesh_seq_examples)])
        label_list.append(y)

    y_list = []

    for y_ in label_list:
        y = np.zeros((len(y_), 2))

        for i in range(len(y_)):
            label = y_[i]

            if label == -1:
                label = 0

            y[i, label] = 1
        y_list.append(y)


    return X_list, y_list, embeddings


def build_embeddings(vocab, w2v, w2v_vector_len):
    embeddings = np.empty((len(vocab) + 1, w2v_vector_len))

    for word, value in vocab.items():
        if word in w2v:
           word_vector = w2v[word]
        else:
            word_vector = np.zeros(w2v_vector_len)

        embeddings[value, :] = word_vector

    return embeddings



def get_data_separately(max_words, word_vector_size, w2v, use_abstract_cnn=False):
    file_paths = get_all_files('Data')
    X_list, y_list = [], []

    total_words = 0.0
    total_words_title = 0.0
    total_mesh_terms = 0.0

    n_abstracts = 0.0

    for file in file_paths:
        data_frame = pd.read_csv(file)

        abstract_text, abstract_labels = extract_abstract_and_labels(data_frame)
        mesh_terms, title = extract_mesh_and_title(data_frame)

        abstracts_as_words = []
        labels = []
        if use_abstract_cnn:
            abstract_mesh_terms = []
            titles = []

        for i in range(abstract_text.shape[0]):
            abstract = abstract_text.iloc[i]

            if abstract == 'MISSING':
                continue
            else:
                if use_abstract_cnn:
                    mesh_term = mesh_terms[i]
                    abstract_title = title[i]
                    preprocessed_mesh_terms = preprocess(mesh_term, tokenize=True)
                    preprocessed_title = preprocess(abstract_title, tokenize=True)

                    total_mesh_terms += len(preprocessed_mesh_terms)
                    total_words_title += len(preprocessed_title)

                    abstract_mesh_terms.append(preprocessed_mesh_terms)
                    titles.append(preprocessed_title)

                preprocessed_abstract = preprocess(abstract, tokenize=True)

                total_words += len(preprocessed_abstract)
                n_abstracts += 1.0

                abstracts_as_words.append(preprocessed_abstract)
                labels.append(abstract_labels.iloc[i])

        X = np.empty((len(abstracts_as_words), 1, max_words['text'], word_vector_size))

        if use_abstract_cnn:
            X_mesh = np.empty((len(abstract_mesh_terms), 1, max_words['mesh'], word_vector_size))
            X_title = np.empty((len(titles), 1, max_words['title'], word_vector_size))

        y = np.zeros((len(labels), 2))

        for i, (abstract, label) in enumerate(zip(abstracts_as_words, labels)):

            word_matrix = text2w2v(abstract, max_words['text'], w2v, word_vector_size)

            if use_abstract_cnn:
                mesh_term_matrix = text2w2v(abstract_mesh_terms[i], max_words['mesh'], w2v, word_vector_size)
                title_matrix = text2w2v(titles[i], max_words['title'], w2v, word_vector_size)

                X_mesh[i, 0, :, :] = mesh_term_matrix
                X_title[i, 0, :, :] = title_matrix

            X[i, 0, :, :] = word_matrix

            if label == -1:
                label = 0
            y[i, label] = 1
        if use_abstract_cnn:
            X_list.append([X, X_title, X_mesh])
        else:
            X_list.append(X)
        y_list.append(y)
    average_word_per_abstract = float(total_words)/float(n_abstracts)
    average_words_per_title = float(total_words_title)/float(n_abstracts)
    average_words_per_mesh = float(total_mesh_terms)/float(n_abstracts)

    print('Average words per abstract: {}'.format(average_word_per_abstract))
    print('Average words per title: {}'.format(average_words_per_title))
    print('Average words per mesh terms: {}'.format(average_words_per_mesh))

    return X_list, y_list


def extract_abstract_and_labels(data_frame):
    abstract_text = data_frame.iloc[:, 4]
    labels = data_frame.iloc[:, 6]

    return abstract_text, labels


def extract_mesh_and_title(data_frame):

    mesh_terms = data_frame.iloc[:, 5]
    titles = data_frame.iloc[:, 1]

    return mesh_terms, titles


def get_data(max_words, word_vector_size, w2v):

    file_paths = get_all_files('Data')

    abstract_text_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    for file in file_paths:
        abstract_text, labels = extract_abstract_and_labels(file)

        abstract_text_df = pd.concat((abstract_text_df, abstract_text))
        labels_df = pd.concat((labels_df, labels))

    abstracts_as_words = []
    labels = []
    total_words = 0

    for i in range(abstract_text_df.shape[0]):
        abstract = abstract_text_df.iloc[i, :][0]

        if abstract == 'MISSING':
            continue
        else:
            abstract_as_words = nltk.word_tokenize(abstract)
            abstracts_as_words.append(abstract_as_words)
            labels.append(labels_df.iloc[i])

            total_words += len(abstracts_as_words)
    X = np.empty((len(abstracts_as_words), 1, max_words , word_vector_size))
    y = np.zeros((len(labels), 2))

    print(total_words/len(abstracts_as_words))

    for i, (abstract, label) in enumerate(zip(abstracts_as_words, labels)):
        word_matrix = text2w2v(abstract, max_words, w2v, word_vector_size)
        X[i, 0, :, :] = word_matrix
        label = label[0]
        if label == -1:
            label = 0
        y[i, label] = 1

    return X, y


def text2w2v(words, max_words, w2v, word_vector_size, remove_stop_words=False):

    if remove_stop_words:
        stop = stopwords.words('english')
    else:
        stop = None
    i = 0
    word_matrix = np.zeros((max_words, word_vector_size))

    for word in words:
        if i == max_words - 1:
            break

        if remove_stop_words:
            if word in stop:
                continue

        if word in w2v:
            word_vector = w2v[word]
            word_matrix[i, :] = word_vector
            i += 1

    return word_matrix[np.newaxis, :, :]


def create_whitespace(length):
    whitespace = ''.join(" " for i in range(length))

    return whitespace


def preprocess(line, tokenize=True, to_lower=True):
    punctuation = "`~!@#$%^&*()_-=+[]{}\|;:'\"|<>,./?åαβ"
    numbers = "1234567890"
    number_replacement = create_whitespace(len(numbers))
    spacing = create_whitespace(len(punctuation))

    if to_lower:
        line = line.lower()

    translation_table = str.maketrans(punctuation, spacing)
    translated_line = line.translate(translation_table)
    translation_table_numbers = str.maketrans(numbers, number_replacement)
    final_line = translated_line.translate(translation_table_numbers)

    if tokenize:
        line_tokens = final_line.split()

        return line_tokens
    else:
        return final_line
