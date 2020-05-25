import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    """
        RNN model for text classification
    """
    def __init__(self, vocab_size, num_class, emb_dim, emb_droprate, rnn_cell_hidden, rnn_cell_type, birnn, num_layers, rnn_droprate, sequence_len):
        super().__init__()
        self.vocab_size = vocab_size                # vocab size
        self.emb_dim = emb_dim                      # embedding dimension
        self.emb_droprate = emb_droprate            # embedding droprate
        self.num_class = num_class                  # classes
        self.rnn_cell_hidden = rnn_cell_hidden      # hidden layer size
        self.rnn_cell_type = rnn_cell_type          # rnn cell type
        self.birnn = birnn                          # wheather use bidirectional rnn
        self.num_layers = num_layers                # number of rnn layers
        self.rnn_droprate = rnn_droprate            # rnn dropout rate before fc
        self.sequence_len = sequence_len            # fix sequence length, so we dont need loop
        pass

    def build(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_dropout = nn.Dropout(self.emb_droprate)
        if self.rnn_cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.emb_dim, hidden_size=self.rnn_cell_hidden, num_layers=self.num_layers, bidirectional=self.birnn, batch_first=True)
        elif self.rnn_cell_type == "GRU":
            self.rnn = nn.GRU(input_size=self.emb_dim, hidden_size=self.rnn_cell_hidden, num_layers=self.num_layers, bidirectional=self.birnn, batch_first=True)
        else:
            self.rnn = None
            print("unsupported rnn cell type, valid is [LSTM, GRU]")
        if self.birnn:
            self.fc = nn.Linear(2 * self.rnn_cell_hidden, self.num_class)
        else:
            self.fc = nn.Linear(self.rnn_cell_hidden, self.num_class)

        self.rnn_dropout = nn.Dropout(self.rnn_droprate)

    def forward(self, input_):
        batch_size = input_.shape[0]

        x = self.embedding(input_)
        x = self.emb_dropout(x)

        if self.rnn_cell_type == "LSTM":
            if self.birnn:
                h_0 = torch.zeros(self.num_layers * 2, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
                c_0 = torch.zeros(self.num_layers * 2, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
            else:
                h_0 = torch.zeros(self.num_layers, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
                c_0 = torch.zeros(self.num_layers, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
            output, (h_n, c_n) = self.rnn(x, (h_0, c_0))
        elif self.rnn_cell_type == "GRU":
            if self.birnn:
                h_0 = torch.zeros(self.num_layers * 2, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
            else:
                h_0 = torch.zeros(self.num_layers, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
            output, h_n = self.rnn(x, h_0)

        if self.birnn:
            x = h_n.view(self.num_layers, 2, batch_size, self.rnn_cell_hidden)
            x = torch.cat((x[-1, 0, : , : ], x[-1, 1, : , : ]), dim = 1)
        else:
            x = h_n.view(self.num_layers, 1, batch_size, self.rnn_cell_hidden)
            x = x[-1, 0, : , : ]

        x = x.view(batch_size, 1, -1)           # shape: [batch_size, 1, 2 or 1 * rnn_cell_hidden]
        x = self.rnn_dropout(x)

        x = self.fc(x)
        x = x.view(-1, self.num_class)          # shape: [batch_size, num_class]

        return x
