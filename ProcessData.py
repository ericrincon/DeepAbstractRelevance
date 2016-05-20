import DataLoader
import sys
import getopt
import h5py
from gensim.models import Word2Vec

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['max_words_abstract=', 'max_words_title=', 'max_words_mesh=',
                                                      'path=', 'w2v_path=', 'w2v_length='])

    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    max_words = {'text': 270, 'mesh': 50, 'title': 17}
    path = 'Data/'
    w2v_path = '/Users/ericrincon/PycharmProjects/Deep-PICO/wikipedia-pubmed-and-PMC-w2v.bin'
    word_vector_size = 200
    filter_small_data = True

    print('Loading word2vec...')
    w2v = Word2Vec.load_word2vec_format(w2v_path, binary=True)
    print('Loaded word2vec...')

    for opt, arg in opts:
        if opt == '--max_words_abstract':
            max_words['text'] = int(arg)
        elif opt == '--max_words_title':
            max_words['mesh'] = int(arg)
        elif opt == '--max_words_mesh':
            max_words['mesh'] = int(arg)
        elif opt == '--path':
            path = arg
        elif opt == '--w2v_path':
            w2v_path = arg

    X_list, y_list, data_names = DataLoader.get_data_separately(max_words, word_vector_size, w2v, use_abstract_cnn=True,
                                                        preprocess_text=False, filter_small_data=filter_small_data)

    for X, y, name in zip(X_list, y_list, data_names):
        X_abstract, X_title, X_mesh = X

        f = h5py.File("DataProcessed/" + name + ".hdf5", "w")
        f.create_dataset('X_abstract', data=X_abstract, shape=X_abstract.shape)
        f.create_dataset('X_title', data=X_title, shape=X_title.shape)
        f.create_dataset('X_mesh', data=X_mesh, shape=X_mesh.shape)
        f.create_dataset('y', data=y, shape=y.shape)

if __name__ == '__main__':
    main()