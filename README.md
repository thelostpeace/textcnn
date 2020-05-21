# Text Classification
use multiple kind of models to do text classification

## TextCNN
TextCNN pytorch implement. paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

```
------------------------------------------------------------------------
      Layer (type)         Output Shape         Param #     Tr. Param #
========================================================================
       Embedding-1        [32, 64, 300]       1,441,500       1,441,500
         Dropout-2        [32, 64, 300]               0               0
          Conv2d-3     [32, 128, 63, 1]          76,928          76,928
          Conv2d-4     [32, 128, 62, 1]         115,328         115,328
          Conv2d-5     [32, 128, 61, 1]         153,728         153,728
       MaxPool2d-6      [32, 128, 1, 1]               0               0
       MaxPool2d-7      [32, 128, 1, 1]               0               0
       MaxPool2d-8      [32, 128, 1, 1]               0               0
         Dropout-9         [32, 1, 384]               0               0
         Linear-10          [32, 1, 27]          10,395          10,395
========================================================================
Total params: 1,797,879
Trainable params: 1,797,879
Non-trainable params: 0
------------------------------------------------------------------------
```

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
