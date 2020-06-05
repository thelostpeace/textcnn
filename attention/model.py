import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    '''
        Luong Attention Mechanism
    '''
    def __init__(self, query_hidden_size, value_hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.query_hidden_size = query_hidden_size
        self.value_hidden_size = value_hidden_size
        self.method = method

        if self.method == "general":
            self.fc = nn.Linear(self.query_hidden_size, self.value_hidden_size, bias=False)

        if self.method == "concat":
            self.fc = nn.Linear(self.query_hidden_size, self.value_hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(self.value_hidden_size, 1))

        if self.method == "genquery":
            self.fc = nn.Linear(self.value_hidden_size, 1, bias=False)

    def forward(self, query, value):
        batch_size = value.shape[0]
        if self.method == "dot":
            return query.bmm(value.transpose(1, 2))    # batch_size * slen

        if self.method == "general":
            out = self.fc(query)
            return out.bmm(value.transpose(1, 2))    # batch_size * slen

        if self.method == "concat":
            out = torch.tanh(self.fc(query + value))
            weight = self.weight.expand(batch_size, self.hidden_size, 1)
            return out.bmm(weight).transpose(1, 2)

        if self.method == "genquery":
            return self.fc(value).transpose(1, 2)


class AttentionBiLSTM(nn.Module):
    """
        RNN model for text classification
    """
    def __init__(self, vocab_size, num_class, emb_dim, emb_droprate, rnn_cell_hidden, num_layers, att_droprate, sequence_len, att_method):
        super().__init__()
        self.vocab_size = vocab_size                # vocab size
        self.emb_dim = emb_dim                      # embedding dimension
        self.emb_droprate = emb_droprate            # embedding droprate
        self.num_class = num_class                  # classes
        self.num_layers = num_layers                # number of rnn layers
        self.att_droprate = att_droprate            # attention dropout rate before fc
        self.sequence_len = sequence_len            # fix sequence length, so we dont need loop
        self.rnn_cell_hidden = rnn_cell_hidden      # LSTM hidden size
        self.att_method = att_method                # dot, general or concat
        #self.attention = Attention(2 * self.rnn_cell_hidden, 2 * self.rnn_cell_hidden, self.att_method)   # attention layer
        self.attention = Attention(2 * self.rnn_cell_hidden, self.emb_dim, self.att_method)   # attention layer
        pass

    def build(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_dropout = nn.Dropout(self.emb_droprate)
        self.rnn = nn.LSTM(input_size=self.emb_dim, hidden_size=self.rnn_cell_hidden, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        #self.fc = nn.Linear(4 * self.rnn_cell_hidden, self.num_class)
        self.fc = nn.Linear(2 * self.rnn_cell_hidden + self.emb_dim, self.num_class)

        self.att_dropout = nn.Dropout(self.att_droprate)

    def forward(self, input_):
        batch_size = input_.shape[0]

        emb = self.embedding(input_)
        x = self.emb_dropout(emb)

        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.rnn_cell_hidden, requires_grad=True).to(device)
        output, (h_n, c_n) = self.rnn(x, (h_0, c_0))

        #value = output.view(batch_size, -1, 2 * self.rnn_cell_hidden)
        value = x
        hidden = h_n.view(self.num_layers, 2, batch_size, self.rnn_cell_hidden)
        hidden = torch.cat((hidden[-1, 0, : , : ], hidden[-1, 1, :, : ]), dim=1)
        query = hidden.view(batch_size, 1, -1)

        align = self.attention(query, value)
        #print("align:", align.shape)
        score = F.softmax(align, dim=2)
        #print(score)
        score = score.transpose(1, 2)
        #print("score:", score.shape)
        value = value.transpose(1, 2)
        #print("value:", value.shape)
        value = torch.bmm(value, score).transpose(1, 2)
        #print("value:", value.shape)
        #print("query:", query.shape)
        h_new = torch.cat((value, query), dim=2)
        #print("hnew:", h_new.shape)

        x = self.att_dropout(h_new)

        x = self.fc(x)
        x = x.view(-1, self.num_class)          # shape: [batch_size, num_class]

        return x
