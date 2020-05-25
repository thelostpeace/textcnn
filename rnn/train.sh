# LSTM 1 layer bidirectional=False test acc: 95.79%
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0001 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --rnn_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 1000 --rnn_cell_type LSTM --birnn False --num_layers 1

# LSTM 1 layer bidirectional=True test acc: 95.8%
#CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.00003 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --rnn_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 500 --rnn_cell_type LSTM --birnn True --num_layers 1

# LSTM 2 layer bidirectional=True test acc:
#CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.00005 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --rnn_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 300 --rnn_cell_type LSTM --birnn True --num_layers 2

# GRU 1 layer bidirectional=False 
#CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.00005 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --rnn_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 1000 --rnn_cell_type GRU --birnn False --num_layers 1

# GRU 1 layer bidirectional=True
#CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.00005 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --rnn_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 500 --rnn_cell_type GRU --birnn True --num_layers 1

# GRU 2 layer bidirectional=True acc: 95.87%
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.00005 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --rnn_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 300 --rnn_cell_type GRU --birnn True --num_layers 2


# valid accuracy are almost the same, 1 layer or 2 layers, GRU or LSTM, bidirectional or not bidirectional, the difference I see is that GRU network is more steady when training. And training for GRU is faster.
