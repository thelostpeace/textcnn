# TextCNN
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
