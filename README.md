# Text Classification
use multiple kind of models to do text classification

## TextCNN
TextCNN pytorch implement. paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

### data
original data is not provided.

```
# training data format
# label\tspace split token
# there is a simple unicode tokenizer in tool/tokenizer.py, if you have large corpus, you can use jieba to do word seg
reminad	7:20 提 醒 我 要 上 学 了
```

### train

```
cd cnn
sh train.sh
```

### predict

online predict part is not written, and I won't write it in the future, because it is easy to write the online part.

## RNN

use one-layer or multi-layer `LSTM`, `biLSTM`, `GRU`, `biGRU` to do text classification.

```
cd rnn
sh train.sh
```

## RNN with Attention

not implete yet.
