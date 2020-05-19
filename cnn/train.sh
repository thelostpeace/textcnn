# overfit
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.001 --batch_size 32 --epoch 100 --filter_count 128 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

# overfit but better than first 1
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.001 --batch_size 32 --epoch 100 --filter_count 64 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

# overfit 
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.001 --batch_size 32 --epoch 100 --filter_count 64 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.6 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

# better than last one for lower learning rate
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0005 --batch_size 32 --epoch 100 --filter_count 64 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.6 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

# add weight decay, but learning is slow, 100 epoch not even fit the training data
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.0001 --batch_size 32 --epoch 100 --filter_count 64 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.00005 --batch_size 32 --epoch 300 --filter_count 64 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.00005 --batch_size 32 --epoch 300 --filter_count 128 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt

CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.00008 --batch_size 32 --epoch 300 --filter_count 128 --seed 1992 --embedding_dim 300 --embedding_droprate 0.5 --sequence_len 64 --kernel_size 2 3 4 --conv_droprate 0.5 --train ../data/train.txt --dev ../data/dev.txt --test ../data/test.txt --mode train --output_vocab_label ./model/class.txt --output_vocab_word ./model/vocab.txt
