import sys
sys.path.insert(0, '../')
from model import BertPretrainClassification
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import time
from tool.data_loader import build_vocab, build_label
from tool.data_loader import load_vocab
from tool.data_loader import TextDataSet
from torch.utils.tensorboard import SummaryWriter
import os, glob, shutil
from pytorch_model_summary import summary
import model_config
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train")
args = parser.parse_args()

config = model_config._model_config

torch.manual_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# clear tensorboard logs
if os.path.exists(config.tensorboard):
    shutil.rmtree(config.tensorboard)
os.mkdir(config.tensorboard)
writer = SummaryWriter(log_dir=config.tensorboard, flush_secs=60)

checkpoint_path = "checkpoints"
if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
os.mkdir(checkpoint_path)

# do text parsing, get vocab size and class count
build_label(config.train, config.vocab_label)
label2id, id2label = load_vocab(config.vocab_label)

num_class = len(label2id)
print("class count: %s" % num_class)
# set model
bert_config = BertConfig.from_json_file(config.bert_config)
bert_config.output_hidden_states = True
model = BertPretrainClassification.from_pretrained(config.bert_model, config=bert_config)
#model = BertForSequenceClassification.from_pretrained(config.bert_model, config=bert_config)
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps)

writer.add_graph(model, [torch.randint(low=0,high=1000, size=(config.batch_size, config.sequence_len), dtype=torch.long).to(device), torch.ones(size=(config.batch_size, config.sequence_len), dtype=torch.long).to(device)])
print(summary(model, torch.randint(low=0,high=1000, size=(config.batch_size, config.sequence_len), dtype=torch.long).to(device), torch.ones(size=(config.batch_size, config.sequence_len), dtype=torch.long).to(device)))

# bert tokenizer
tokenizer = BertTokenizer(vocab_file=config.vocab_file, do_lower_case=True)

def generate_batch(batch):
    # TextDataSet yield one line contain label and input
    batch_label, input_ids, attention_masks = [], [], []
    for data in batch:
        num_input = []
        label, sentence = data.split('\t')
        batch_label.append(label2id[label])
        encoded_dict = tokenizer.encode_plus(sentence, max_length=config.sequence_len, pad_to_max_length=True, return_tensors='pt')
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    tensor_label = torch.tensor(batch_label, dtype=torch.long)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return tensor_label.to(device), input_ids.to(device), attention_masks.to(device)

def save_checkpoint(state, is_best, filename="checkpoint"):
    name = "%s_epoch:%s_validacc:%s.pt" % (filename, state['epoch'], state['valid_acc'])
    torch.save(state, "%s/%s" % (checkpoint_path, name))
    if is_best:
        shutil.copyfile("%s/%s" % (checkpoint_path, name), "%s/%s" % (args.save_model, model_name))

def save_model():
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(config.save_model)
    tokenizer.save_pretrained(config.save_model)

def test(test_data):
    valid_loss = 0
    valid_acc = 0
    test_dataset = TextDataSet(test_data)
    data = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=generate_batch)
    model.eval()
    for i, (label, input_ids, attention_mask) in enumerate(data):
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == label).sum().item()

    return valid_loss / len(test_dataset), valid_acc / len(test_dataset)

def train():
    global model
    # calc total steps before set scheduler
    scheduler_dataset = TextDataSet(config.train)
    scheduler_dataloader = DataLoader(scheduler_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=generate_batch)
    total_steps = config.epoch * len(scheduler_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_valid_acc = 0.0
    model.train()
    for epoch in range(config.epoch):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_count = 0
        train_dataset = TextDataSet(config.train)
        data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch)
        for i, (label, input_ids, attention_masks) in enumerate(data):
            optimizer.zero_grad()
            output = model(input_ids, attention_masks)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_acc += (output.argmax(1) == label).sum().item()
            train_count += input_ids.shape[0]
            step_accuracy = train_acc / train_count

            if i % config.save_step == 0:
                print("Epoch:%s Steps:%s train_count:%s" % (epoch, i, train_count))
                valid_loss, valid_acc = test(config.dev)
                if valid_acc > best_valid_acc and epoch > 0:
                    save_model()
                    best_valid_acc = valid_acc

                secs = int(time.time() - start_time)
                mins = secs / 60
                secs = mins % 60
                writer.add_scalars("Loss", {
                    'train': train_loss / train_count,
                    "valid": valid_loss
                    }, epoch * total_steps + i)
                writer.add_scalars("Acc", {
                    'train': train_acc / train_count,
                    "valid": valid_acc
                    }, epoch * total_steps + i)

                print("Epoch: %d" % (epoch + 1), "Steps: %d" % (epoch * total_steps + i), " | time in %d minutes, %d seconds" % (mins, secs))
                print(f"\tLoss: {train_loss / train_count:.4f}(train)\t|\tAcc: {train_acc / train_count * 100:.1f}%(train)")
                print(f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)")

    # test
    model_path = "model/pytorch_model.bin"
    model = BertPretrainClassification.from_pretrained(model_path, config=bert_config)
    model.to(device)
    test_loss, test_acc = test(config.test)
    print(f"\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)")


if __name__ == "__main__":
    if args.mode == 'train':
        train()
