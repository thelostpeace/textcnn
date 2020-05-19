import sys
sys.path.insert(0, '../')
from model import TextCNN
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import time
from tool.data_loader import build_vocab
from tool.data_loader import load_vocab
from tool.data_loader import TextDataSet
from torch.utils.tensorboard import SummaryWriter
import os, glob, shutil

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--filter_count', type=int, default=128)
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--embedding_droprate', type=float, default=0.5)
parser.add_argument('--sequence_len', type=int, default=64)
parser.add_argument('--kernel_size', type=list, nargs='+', default=[2,3,4])
parser.add_argument('--conv_droprate', type=float, default=0.5)
parser.add_argument('--train', type=str, default='train.txt')
parser.add_argument('--dev', type=str, default='dev.txt')
parser.add_argument('--test', type=str, default='test.txt')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--output_vocab_label', type=str, default='./model/class.txt')
parser.add_argument('--output_vocab_word', type=str, default='./model/vocab.txt')
parser.add_argument('--tensorboard', type=str, default='tensorboard')
parser.add_argument('--save_model', type=str, default='./model')
args = parser.parse_args()
kernel_size = [int(k[0]) for k in args.kernel_size]

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# clear tensorboard logs
if os.path.exists(args.tensorboard):
    shutil.rmtree(args.tensorboard)
os.mkdir(args.tensorboard)
writer = SummaryWriter(log_dir=args.tensorboard, flush_secs=60)

checkpoint_path = "checkpoints"
if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
os.mkdir(checkpoint_path)
model_name = "textcnn.pt"


# set model
model = TextCNN(emb_dim=args.embedding_dim, emb_droprate=args.embedding_droprate, seq_len=args.sequence_len, filter_count=args.filter_count, kernel_size=kernel_size, conv_droprate=args.conv_droprate)

# padding sequence with <PAD>
def padding(data, fix_length, pad, add_first="", add_last=""):
    if add_first:
        data.insert(0, add_first)
    if add_last:
        data.append(add_last)
    pad_data = []
    data_len = len(data)
    for idx in range(fix_length):
        if idx < data_len:
            pad_data.append(data[idx])
        else:
            pad_data.append(pad)
    return pad_data

def generate_batch(batch):
    # TextDataSet yield one line contain label and input
    batch_label, batch_input = [], []
    for data in batch:
        num_input = []
        label, sentence = data.split('\t')
        batch_label.append(label2id[label])
        words = sentence.split()
        # pad to fix length
        words = padding(words, args.sequence_len, '<PAD>', add_first='<BOS>', add_last='<EOS>')
        for w in words:
            if w in word2id:
                num_input.append(word2id[w])
            else:
                num_input.append(word2id['<UNK>'])
        batch_input.append(num_input)

    tensor_label = torch.tensor(batch_label, dtype=torch.long)
    tensor_input = torch.tensor(batch_input, dtype=torch.long)

    return tensor_label.to(device), tensor_input.to(device)

def save_checkpoint(state, is_best, filename="checkpoint"):
    name = "%s_epoch:%s_validacc:%s.pt" % (filename, state['epoch'], state['valid_acc'])
    torch.save(state, "%s/%s" % (checkpoint_path, name))
    if is_best:
        shutil.copyfile("%s/%s" % (checkpoint_path, name), "%s/%s" % (args.save_model, model_name))

def train_epoch(train_data):
    train_loss = 0
    train_acc = 0
    train_dataset = TextDataSet(train_data)
    data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=generate_batch)
    for i, (label, inp) in enumerate(data):
        optimizer.zero_grad()
        output = model(inp)
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()

    return train_loss / len(train_dataset), train_acc / len(train_dataset)

def test(test_data):
    valid_loss = 0
    valid_acc = 0
    test_dataset = TextDataSet(test_data)
    data = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=generate_batch)
    for i, (label, inp) in enumerate(data):
        with torch.no_grad():
            output = model(inp)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == label).sum().item()

    return valid_loss / len(test_dataset), valid_acc / len(test_dataset)

def train():
    # do text parsing, get vocab size and class count
    build_vocab(args.train, args.output_vocab_label, args.output_vocab_word)
    label2id, id2label = load_vocab(args.output_vocab_label)
    word2id, id2word = load_vocab(args.output_vocab_word)

    vocab_size = len(word2id)
    num_class = len(label2id)
    model.build()
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    writer.add_graph(model, torch.randint(low=0,high=1000, size=(args.batch_size, args.sequence_len), dtype=torch.long).to(device))


if __name__ == "__main__":
    if args.mode == 'train':
        best_valid_acc = 0.0
        for epoch in range(args.epoch):
            start_time = time.time()
            train_loss, train_acc = train(args.train)
            valid_loss, valid_acc = test(args.dev)

            # save best model
            if valid_acc > best_valid_acc:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'valid_acc': valid_acc
                    }, True)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            writer.add_scalars("Loss", {
                'train': train_loss,
                'valid': valid_loss
                }, epoch)
            writer.add_scalars("Acc", {
                'train': train_acc,
                'valid': valid_acc
                }, epoch)

            print("Epoch: %d" % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f"\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)")
            print(f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)")

        # test
        saved_params = torch.load("%s/%s" % (args.save_model, model_name))
        print("epoch:%s best_valid_acc:%s" % (saved_params['epoch'], saved_params['valid_acc']))
        model.load_state_dict(saved_params['state_dict'])
        loss, acc = test(args.test)
        print("test set loss: %s" % loss)
        print("test set acc: %s" % acc)
