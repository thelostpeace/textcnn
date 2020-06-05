# LSTM 2 layers bidirectional=True attention:dot acc: 96%
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0001 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --att_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 200 --num_layers 2 --att_method dot

# attention: general acc: 95.8%
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0001 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --att_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 200 --num_layers 2 --att_method general

# attention: concat acc: 96.0% 
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0001 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --att_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 200 --num_layers 2 --att_method concat

# attention: genquery acc: 95.9%
CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0001 --batch_size 32 --epoch 100 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --att_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt --rnn_cell_hidden 200 --num_layers 2 --att_method genquery

# attention on word embedding method: general  acc: 95.9%

# attention on word embedding method: genquery acc: 95.9%
