import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """
        TextCNN model
    """
    def __init__(self, emb_dim, emb_droprate, seq_len, filter_count, kernel_size, conv_droprate):
        super().__init__()
        #self.vocab_size = vocab_size        # vocab size
        self.emb_dim = emb_dim              # embedding dimension
        self.emb_droprate = emb_droprate    # embedding dropout rate
        self.seq_len = seq_len              # sequence length
        self.filter_count = filter_count    # output feature size
        self.kernel_size = kernel_size      # list of kernel size, means kGram in text, ex. [1, 2, 3, 4, 5 ...]
        self.conv_droprate = conv_droprate  # conventional layer dropout rate
        #self.num_class = num_class          # classes
        pass

    def build(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_dropout = nn.Dropout(self.emb_droprate)
        self.conv1 = nn.Conv2d(1, self.filter_count, (self.kernel_size[0], self.emb_dim))
        self.conv2 = nn.Conv2d(1, self.filter_count, (self.kernel_size[1], self.emb_dim))
        self.conv3 = nn.Conv2d(1, self.filter_count, (self.kernel_size[2], self.emb_dim))
        self.pool1 = nn.MaxPool2d((self.seq_len - self.kernel_size[0] + 1, 1))
        self.pool2 = nn.MaxPool2d((self.seq_len - self.kernel_size[1] + 1, 1))
        self.pool3 = nn.MaxPool2d((self.seq_len - self.kernel_size[2] + 1, 1))
        self.conv_dropout = nn.Dropout(self.conv_droprate)
        self.fc = nn.Linear(3 * self.filter_count, self.num_class)
        pass

    def forward(self, input_):
        batch_size = input_.shape[0]

        x = self.embedding(input_)
        x = self.emb_dropout(x)

        x = x.view(batch_size, 1, x.shape[1], x.shape[2])
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(batch_size, 1, -1)           # shape: [batch_size, 1, filter_count * conv_layer_count]
        x = self.conv_dropout(x)

        x = self.fc(x)
        x = x.view(-1, self.num_class)          # shape: [batch_size, num_class]

        return x
