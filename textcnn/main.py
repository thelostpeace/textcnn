import sys
from model import TextCNN, Config
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import time
from tools.data_loader import build_vocab, build_label, save_vocab
from tools.data_loader import load_vocab
from tools.data_loader import TextDataSet
from torch.utils.tensorboard import SummaryWriter
import os, glob, shutil, json
from pytorch_model_summary import summary
import numpy as np
from transformers import BertTokenizer
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--config', type=str, default="model.config")
args = parser.parse_args()

config = Config(json.load(open(args.config)))

# load bert pretraining vector and vocab
bert_embedding = np.load(config.bert_pretrain)
save_vocab(bert_embedding['unicode_vocab'], config.vocab_save)
bert_tokenizer = BertTokenizer(vocab_file=config.vocab_save, do_lower_case=True)
build_label(config.train, config.label_save)
label2id, id2label = load_vocab(config.label_save)
extend_config = {
    "bert_embedding": bert_embedding,
    "labels": len(label2id)
}
config.set(extend_config)

torch.manual_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# clear tensorboard logs
if os.path.exists(config.tensorboard):
    shutil.rmtree(config.tensorboard)
os.mkdir(config.tensorboard)
writer = SummaryWriter(log_dir=config.tensorboard, flush_secs=int(config.tensorboard_flush_sec))

checkpoint_path = "checkpoints"
if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
os.mkdir(checkpoint_path)

#print(config)

# set model
model = TextCNN(config)
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)

writer.add_graph(model, torch.randint(low=0,high=100, size=(config.batch_size, 32), dtype=torch.long).cuda())
print(summary(model, torch.randint(low=0,high=100, size=(config.batch_size, 32), dtype=torch.long).cuda()))

def generate_batch(batch):
    # TextDataSet yield one line contain label and input
    batch_label, batch_input = [], []
    for data in batch:
        num_input = []
        if len(data.split('\t')) != 2:
            print(data)
        label, sent = data.split('\t')
        #print(sent1, sent2, label)
        batch_label.append(label2id[label])
        sent = torch.squeeze(bert_tokenizer.encode(bert_tokenizer.tokenize(sent), add_special_tokens=False, return_tensors='pt', max_length=config.sequence_length, pad_to_max_length=True), dim=0).tolist()
        batch_input.append(sent)

    return torch.tensor(batch_input, dtype=torch.long).cuda(), torch.tensor(batch_label, dtype=torch.long).cuda()

def save_checkpoint(state, is_best=True, filename="checkpoint"):
    name = "%s_epoch:%s_steps:%s_validacc:%s.pt" % (filename, state['epoch'], state['steps'], state['valid_acc'])
    torch.save(state, "%s/%s" % (checkpoint_path, name))
    if is_best:
        shutil.copyfile("%s/%s" % (checkpoint_path, name), "%s" % (config.save_model))

def test(test_data):
    valid_loss = 0
    valid_acc = 0
    test_dataset = TextDataSet(test_data)
    data = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=generate_batch)
    #model.eval()
    y_test, y_pred = [], []
    for i, (input_, label) in enumerate(data):
        with torch.no_grad():
            output, _ = model(input_)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == label).sum().item()

            y_test.extend(label.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())
    labels = list(id2label.keys())
    cm = confusion_matrix(y_test, y_pred, labels)

    return valid_loss / len(test_dataset), valid_acc / len(test_dataset), cm

def train():
    global model

    best_valid_acc = 0.0
    steps = 0
    saved_step = 0
    total_steps = 0
    stop_train = False
    validation_flag = False
    for epoch in range(config.epoch):
        start_time = time.time()
        train_loss = 0
        train_count = 0
        train_acc = 0
        train_dataset = TextDataSet(config.train)
        data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch)
        for i, (input_, label) in enumerate(data):
            #model.train()
            optimizer.zero_grad()
            output, _ = model(input_)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad == True], 1.0)
            optimizer.step()
            train_acc += (output.argmax(1) == label).sum().item()
            train_count += input_.shape[0]
            step_accuracy = train_acc / train_count

            if steps % config.save_step == 0 and validation_flag == True:
                total_steps += 1
                #print("Epoch:%s Steps:%s train_count:%s" % (epoch, steps, train_count))
                valid_loss, valid_acc, _ = test(config.dev)
                if valid_acc > best_valid_acc:
                    save_checkpoint({
                        "epoch": epoch + 1,
                        "steps": steps,
                        "state_dict": model.state_dict(),
                        "valid_acc": valid_acc
                        })
                    best_valid_acc = valid_acc
                    saved_step = total_steps

                secs = int(time.time() - start_time)
                mins = secs / 60
                secs = mins % 60
                writer.add_scalars("StepLoss", {
                    'train': train_loss / train_count,
                    "valid": valid_loss
                    }, steps)
                writer.add_scalars("StepAcc", {
                    'train': train_acc / train_count,
                    "valid": valid_acc
                    }, steps)

                print("Epoch: %d" % (epoch + 1), "Steps: %d" % steps, " | time in %d minutes, %d seconds" % (mins, secs))
                print(f"\tLoss: {train_loss / train_count:.4f}(train)\t|\tAcc: {train_acc / train_count * 100:.1f}%(train)")
                print(f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)")

                if total_steps - saved_step > config.early_stop and validation_flag == True:
                    stop_train = True
                    break
            steps += 1

        if stop_train == True:
            print("early stop at epoch %s!!!" % (epoch + 1))
            break

        if train_acc / len(train_dataset) > config.val_train_acc:
            validation_flag = True
        # tensorboard for epoch accuracy and loss
        writer.add_scalars("EpochLoss", {
            'train': train_loss / len(train_dataset),
            }, epoch + 1)
        writer.add_scalars("EpochAcc", {
            'train': train_acc / len(train_dataset),
            }, epoch + 1)

def print_confusion_matrix(cm, labels):
    def format_row(row):
        width = 10
        ret = ''
        for r in row:
            ret += '{val:{width}}'.format(val=str(r), width=width)
        return ret
    title = [' ']
    title.extend(labels)
    print(format_row(title))
    for i in range(len(labels)):
        row = [labels[i]]
        row.extend(cm[i])
        print(format_row(row))

def evaluate():
    # test
    model = TextCNN(config)
    model.cuda()
    saved_model = torch.load(config.save_model)
    model.load_state_dict(saved_model["state_dict"])
    print("epoch:%s steps:%s best_valid_acc:%s" % (saved_model["epoch"], saved_model["steps"], saved_model["valid_acc"]))

    test_loss, test_acc, cm = test(config.test)
    print(f"\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)")

    print_confusion_matrix(cm, list(id2label.values()))


if __name__ == "__main__":
    if args.mode == 'train':
        train()
        evaluate()
    if args.mode == 'pretrain':
        train()
    if args.mode == 'test':
        evaluate()
