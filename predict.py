from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import sys
import pre_processing
import final_predict
def main():
    # create instance of config

    pre_processing.pre_processing_test(input_file)
    config = Config()
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    predict=CoNLLDataset("mytest.txt", config.processing_word, config.max_iter)
    max_sequence_length = max(max([len(seq[0]) for seq in train]), max([len(seq[0]) for seq in dev]),
                              max([len(seq[0]) for seq in test]))

    model = NERModel(config,max_sequence_length)
    model.build()
    model.restore_session(config.dir_model)
    model.run_predict(predict)






if __name__ == "__main__":
    main()
