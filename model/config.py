import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):

        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)

        self.nwords     = len(self.vocab_words)
        self.ntags      = len(self.vocab_tags)


        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words)
        self.processing_tag  = get_processing_word(self.vocab_tags,allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300

    # glove files
    filename_glove = "data/word2vec.40B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/word2vec.40B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = "data/mydev.txt"
    filename_test = "data/mydev.txt"
    filename_train = "data/mytrain.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"

    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.8
    batch_size       = 1
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1
    nepoch_no_imprv  = 3
    hidden_size_lstm = 300


    use_crf = True

