import numpy as np
import pandas as pd
import nltk
import h5py
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

def load_dataset_from_h5py(path, load_mesh_title, load_as_np):
    data_frame = h5py.File(path, 'r')

    X = {}

    X_abstract = data_frame['X_abstract']
    y = data_frame['y']

    if load_mesh_title:
        X_title = data_frame['X_title']
        X_mesh = data_frame['X_mesh']

    if load_as_np:
        X['text'] = X_abstract[()]

        if load_mesh_title:
            X['title'] = X_title[()]
            X['mesh'] = X_mesh[()]
    else:
        X['text'] = X_abstract

        if load_mesh_title:
            X['title'] = X_title
            X['mesh'] = X_mesh

    return X, y


def load_datasets_from_h5py(path, load_mesh_title=True, load_as_np=False):
    files = get_all_files(path)
    X_list = []
    y_list = []

    for file in files:
        X, y = load_dataset_from_h5py(file, load_mesh_title, load_as_np)

        X_list.append(X)
        y_list.append(y)

    return X_list, y_list

def load_datasets_from_h5py_for_class(path, max_words, w2v_length, save=True):
    X_list, y_list = load_datasets_from_h5py(path, load_mesh_title=False, load_as_np=False)

    n_examples = 0
    start = 0


    for _X in X_list:
        _X = _X['text']
        n_examples += _X.shape[0]

    X = np.empty((n_examples, max_words, w2v_length))
    y = np.empty((n_examples, 2))
    domain_embeddings = np.empty((n_examples, 1))

    for i, (_X, __y) in enumerate(zip(X_list, y_list)):
        _X = _X['text']

        neg = np.where(__y[:, 0] == 1)[0]
        pos = np.where(__y[:, 1] == 1)[0]

        _y = np.zeros((_X.shape[0], 2))
        _y[neg, 0] = 1
        _y[pos, 1] = 1

        domain_embeddings[start: start + _X.shape[0], 0] = i

        X[start: start + _X.shape[0], :] = _X
        y[start: start + _X.shape[0], :] = _y
        start += _X.shape[0]
    if save:
        df = h5py.File('all_domains.hdf5', 'w')
        df.create_dataset('X', data=X, shape=X.shape)
        df.create_dataset('y', data=y, shape=y.shape)
        df.create_dataset('de', data=domain_embeddings, shape=domain_embeddings.shape)
    else:
        return X, y



def texts2seq(texts, sizes, vocab=None, w2v=None, w2v_size=200):
    if type(texts) is not list:
        texts = [texts]

    tokenized_texts = []
    seqs = []

    if vocab is not None:
        vocab_size = len(vocab)
        # Add parameter PADDING to vocab????
        vocab['PADDING'] = vocab_size
        vocab_size += 1
    else:
        assert w2v is not None, "Pass a word2vec model!!!!!"


    for text in texts:
        tokenized_texts.append(nltk.tokenize.word_tokenize(text))

    for i, tokenized_text in enumerate(tokenized_texts):
        max_size = sizes[i]

        seq = []

        for token in tokenized_text:
            if vocab is not None:
                if token in vocab:
                    seq.append(vocab[token])
                else:
                    vocab[token] = vocab_size
                    seq.append(vocab_size)
                    vocab_size += 1
            else:
                if token in w2v:
                    word_vector = w2v[token]
                else:
                    word_vector = np.zeros(w2v_size)

                seq.append(word_vector)

        n = len(seq)

        if not n == max_size:
            padding_size = max_size - n

            padding_vector = [np.zeros(w2v_size) for j in range(padding_size)]

            seq = seq + padding_vector

        seqs.append(np.array(seq))

    return seqs


def get_data_as_seq_():
    file_paths = get_all_files('Data')

    if acnn:
        sizes = [max_words['text'], max_words['mesh'], max_words['title']]
    else:
        sizes = [max_words['text']]

    X_list, label_list = [], []
    embeddings = []
    n_examples = 0

    for file_path in file_paths:
        data_frame = pd.read_csv(file_path)

        abstract_text, abstract_labels = extract_abstract_and_labels(data_frame)
        mesh_terms, title = extract_mesh_and_title(data_frame)
        if use_embedding:
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

            if not acnn:
                text_seq = texts2seq([abstract], sizes, vocab=None, w2v=w2v, w2v_size=200)
            else:
                text_seq, mesh_seq, title_seq = texts2seq([abstract, mesh, abstract_title], sizes)

            text_seq_examples.extend(np.array(text_seq))

            if acnn:
                title_seq_examples.extend(title_seq)
                mesh_seq_examples.append(np.array(mesh_seq))


        if acnn:
            embedding = build_embeddings(vocab_dict, w2v, w2v_vector_len)
            embeddings.append(embedding)

            X_list.append([np.array(text_seq_examples), np.array(title_seq_examples), np.array(mesh_seq_examples)])
        else:
            X_list.append(text_seq_examples)
        label_list.append(labels)

    y_list = []

    for y_ in label_list:
        y = np.zeros((len(y_), 2))

        for i in range(len(y_)):
            label = y_[i]

            if label == -1:
                label = 0

            y[i, label] = 1
        y_list.append(y)


    if embeddings:
        return X_list, y_list, embeddings
    else:
        return X_list, y_list


def get_data_as_seq(w2v, w2v_vector_len, max_words, use_embedding=False, acnn=False):
    file_paths = get_all_files('Data')

    if acnn:
        sizes = [max_words['text'], max_words['mesh'], max_words['title']]
    else:
        sizes = [max_words['text']]

    X_list, label_list = [], []
    embeddings = []
    n_examples = 0

    for file_path in file_paths:
        data_frame = pd.read_csv(file_path)

        abstract_text, abstract_labels = extract_abstract_and_labels(data_frame)
        mesh_terms, title = extract_mesh_and_title(data_frame)
        if use_embedding:
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

            if not acnn:
                text_seq = texts2seq([abstract], sizes, vocab=None, w2v=w2v, w2v_size=200)
            else:
                text_seq, mesh_seq, title_seq = texts2seq([abstract, mesh, abstract_title], sizes)

            text_seq_examples.append([np.array(text_seq)])

            if acnn:
                title_seq_examples.extend(title_seq)
                mesh_seq_examples.append(np.array(mesh_seq))


        if acnn:
            embedding = build_embeddings(vocab_dict, w2v, w2v_vector_len)
            embeddings.append(embedding)

            X_list.append([np.array(text_seq_examples), np.array(title_seq_examples), np.array(mesh_seq_examples)])
        else:
            X_list.append(text_seq_examples)
        label_list.append(labels)

    y_list = []

    for y_ in label_list:
        y = np.zeros((len(y_), 2))

        for i in range(len(y_)):
            label = y_[i]

            if label == -1:
                label = 0

            y[i, label] = 1
        y_list.append(y)


    if embeddings:
        return X_list, y_list, embeddings
    else:
        return X_list, y_list


def build_embeddings(vocab, w2v, w2v_vector_len):
    embeddings = np.empty((len(vocab) + 1, w2v_vector_len))

    for word, value in vocab.items():
        if word in w2v:
           word_vector = w2v[word]
        else:
            word_vector = np.zeros(w2v_vector_len)

        embeddings[value, :] = word_vector

    return embeddings


def get_data_separately(max_words, word_vector_size, w2v, use_abstract_cnn=False, preprocess_text=False,
                        filter_missing=True, filter_small_data=True):
    file_paths = get_all_files('Data')
    X_list, y_list = [], []

    total_words = 0.0
    total_words_title = 0.0
    total_mesh_terms = 0.0

    n_abstracts = 0.0
    names = []

    for file in file_paths:
        data_frame = pd.read_csv(file)
        names.append(file.split('/')[1].split('.')[0])

        if data_frame.shape[0] < 800:
            continue

        abstract_text, abstract_labels = extract_abstract_and_labels(data_frame)
        mesh_terms, title = extract_mesh_and_title(data_frame)

        abstracts_as_words = []
        labels = []

        if use_abstract_cnn:
            abstract_mesh_terms = []
            titles = []

        for i in range(abstract_text.shape[0]):
            abstract = abstract_text.iloc[i]

            if filter_missing and abstract == 'MISSING':
                continue
            else:
                if use_abstract_cnn:
                    mesh_term = mesh_terms[i]
                    abstract_title = title[i]

                    if preprocess_text:
                        mesh_term = preprocess(mesh_term, tokenize=True)
                        abstract_title = preprocess(abstract_title, tokenize=True)
                    else:
                        mesh_term = nltk.word_tokenize(mesh_term)
                        abstract_title = nltk.word_tokenize(abstract_title)

                    total_mesh_terms += len(mesh_term)
                    total_words_title += len(abstract_title)

                    abstract_mesh_terms.append(mesh_term)
                    titles.append(abstract_title)

                if preprocess_text:
                    abstract = preprocess(abstract, tokenize=True)
                else:
                    abstract = nltk.word_tokenize(abstract)

                total_words += len(abstract)
                n_abstracts += 1.0

                abstracts_as_words.append(abstract)
                labels.append(abstract_labels.iloc[i])

        X = np.empty((len(abstracts_as_words), max_words['text'], word_vector_size))

        if use_abstract_cnn:
            X_mesh = np.empty((len(abstract_mesh_terms),  max_words['mesh'], word_vector_size))
            X_title = np.empty((len(titles),  max_words['title'], word_vector_size))

        y = np.zeros((len(labels), 2))

        for i, (abstract, label) in enumerate(zip(abstracts_as_words, labels)):

            word_matrix = text2w2v(abstract, max_words['text'], w2v, word_vector_size)

            if use_abstract_cnn:
                mesh_term_matrix = text2w2v(abstract_mesh_terms[i], max_words['mesh'], w2v, word_vector_size)
                title_matrix = text2w2v(titles[i], max_words['title'], w2v, word_vector_size)

                X_mesh[i, :, :] = mesh_term_matrix
                X_title[i, :, :] = title_matrix

            X[i, :, :] = word_matrix

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

    return X_list, y_list, names


def extract_abstract_and_labels(data_frame):
    abstract_text = data_frame.iloc[:, 4]
    labels = data_frame.iloc[:, 6]

    return abstract_text, labels


def extract_mesh_and_title(data_frame):

    mesh_terms = data_frame.iloc[:, 5]
    titles = data_frame.iloc[:, 1]

    return mesh_terms, titles


def get_data(max_words, word_vector_size, w2v, use_abstract_cnn, preprocess_text):

    file_paths = get_all_files('Data')

    abstract_text_df = pd.DataFrame()
    mesh_df = pd.DataFrame()
    title_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    for file in file_paths:
        data_frame = pd.read_csv(file)

        abstract_text, abstract_labels = extract_abstract_and_labels(data_frame)
        mesh_terms, title = extract_mesh_and_title(data_frame)

        abstract_text_df = pd.concat((abstract_text_df, abstract_text))
        mesh_df = pd.concat((mesh_df, mesh_terms))
        title_df = pd.concat((title_df, title))
        labels_df = pd.concat((labels_df, abstract_labels))


    abstract_text_df = abstract_text_df.values
    mesh_df = mesh_df.values
    title_df = title_df.values
    labels_df = labels_df.values

    n = abstract_text_df.shape[0]
    X = np.empty((n, max_words['text'], word_vector_size))
    y = np.zeros((n, 2))


    if use_abstract_cnn:
        X_mesh = np.empty((n,  max_words['mesh'], word_vector_size))
        X_title = np.empty((n,  max_words['title'], word_vector_size))

    for i in range(n):
        abstract = abstract_text_df[i][0]
        label = labels_df[i][0]

        if abstract == 'MISSING':
            continue
        else:
            if use_abstract_cnn:
                mesh_term = mesh_df[i][0]
                abstract_title = title_df[i][0]

                if preprocess_text:
                    mesh_term = preprocess(mesh_term, tokenize=True)
                    abstract_title = preprocess(abstract_title, tokenize=True)
                else:
                    mesh_term = nltk.word_tokenize(mesh_term)
                    abstract_title = nltk.word_tokenize(abstract_title)

            if preprocess_text:
                abstract = preprocess(abstract, tokenize=True)
            else:
                abstract = nltk.word_tokenize(abstract)

        word_matrix = text2w2v(abstract, max_words['text'], w2v, word_vector_size)

        if use_abstract_cnn:
            mesh_term_matrix = text2w2v(mesh_term, max_words['mesh'], w2v, word_vector_size)
            title_matrix = text2w2v(abstract_title, max_words['title'], w2v, word_vector_size)

            X_mesh[i, :, :] = mesh_term_matrix
            X_title[i, :, :] = title_matrix

        X[i, :, :] = word_matrix

        if label == -1:
            label = 0
        y[i, label] = 1

    if use_abstract_cnn:
        return X, X_title, X_mesh, y
    else:
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
        line_tokens = nltk.word_tokenize(final_line)

        return line_tokens
    else:
        return final_line


def slice_seq(seq, indices):
    return [seq[index] for index in indices]


def undersample_acnn(X_abstract_train, X_titles_train, X_mesh_train, y_train):
    # Get all the targets that are not relevant i.e., y = 0
    idx_undersample = np.where(y_train[:, 0] == 1)[0]

    # Get all the targets that are relevant i.e., y = 1
    idx_positive = np.where(y_train[:, 1] == 1)[0]

    # Now sample from the no relevant targets
    random_negative_sample = np.random.choice(idx_undersample, idx_positive.shape[0])

    X_abstract_train_positive = X_abstract_train[idx_positive, :, :]
    X_titles_train_positive = X_titles_train[idx_positive, :, :]
    X_mesh_train_positive = X_mesh_train[idx_positive, :, :]

    X_abstract_train_negative = X_abstract_train[random_negative_sample, :, :]
    X_titles_train_negative = X_titles_train[random_negative_sample, :, :]
    X_mesh_train_negative = X_mesh_train[random_negative_sample, :, :]

    X_abstract_train = np.vstack((X_abstract_train_positive, X_abstract_train_negative))
    X_titles_train = np.vstack((X_titles_train_positive, X_titles_train_negative))
    X_mesh_train = np.vstack((X_mesh_train_positive, X_mesh_train_negative))

    y_train_positive = y_train[idx_positive, :]
    y_train_negative = y_train[random_negative_sample, :]
    y_train = np.vstack((y_train_positive, y_train_negative))

    return X_abstract_train, X_titles_train, X_mesh_train, y_train


def undersample_seq(X_abstract_train, X_titles_train, X_mesh_train, y_train):
    # Get all the targets that are not relevant i.e., y = 0
    idx_undersample = np.where(y_train[:, 0] == 1)[0]

    # Get all the targets that are relevant i.e., y = 1
    idx_positive = np.where(y_train[:, 1] == 1)[0]

    # Now sample from the no relevant targets
    random_negative_sample = np.random.choice(idx_undersample, idx_positive.shape[0])

    X_abstract_train_positive = X_abstract_train[idx_positive]
    X_titles_train_positive = X_titles_train[idx_positive]
    X_mesh_train_positive = X_mesh_train[idx_positive]

    X_abstract_train_negative = X_abstract_train[random_negative_sample]
    X_titles_train_negative = X_titles_train[random_negative_sample]
    X_mesh_train_negative = X_mesh_train[random_negative_sample]

    X_abstract_train = np.hstack((X_abstract_train_positive, X_abstract_train_negative))
    X_titles_train = np.hstack((X_titles_train_positive, X_titles_train_negative))
    X_mesh_train = np.hstack((X_mesh_train_positive, X_mesh_train_negative))

    y_train_positive = y_train[idx_positive, :]
    y_train_negative = y_train[random_negative_sample, :]
    y_train = np.vstack((y_train_positive, y_train_negative))

    return X_abstract_train, X_titles_train, X_mesh_train, y_train

def onehot2list(y):
    return np.argmax(y, axis=1)