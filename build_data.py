from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, \
    get_glove_vocab, write_vocab, load_vocab,  \
    export_trimmed_glove_vectors, get_processing_word


def main():
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word()
    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = [i for i in vocab_words if i in vocab_glove]
    vocab.append(UNK)
    vocab.append("$pad$")
    vocab_tags.append("$pad$")

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,config.filename_trimmed, config.dim_word)




if __name__ == "__main__":
    main()
