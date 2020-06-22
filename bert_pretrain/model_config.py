
config = {
    "lr": 3e-5,
    "seed": 1992,
    "batch_size": 32,
    "epoch": 8,
    "sequence_len": 64,
    "train": "../data/bert_train.txt",
    "dev": "../data/bert_dev.txt",
    "test": "../data/bert_test.txt",
    "vocab_label": "model/class.txt",
    "tensorboard": "tensorboard",
    "save_model": "./model",
    "bert_config": "./bert_model_pt/bert_config.json",
    "bert_model": "./bert_model_pt/pytorch_model.bin",
    "eps": 1e-8,
    "vocab_file": "./bert_model_pt/vocab.txt",
    "save_step": 1000,
    "early_stop_step": 20
}


class Config():
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

            for key in kwargs:
                setattr(self, key, kwargs[key])


_model_config = Config(config)
