import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_layers=1, dropout=0.5, pretrained_embeddings=None):
        super(SimpleRNN, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        hidden = hidden[-1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, 
                 num_layers=2, dropout=0.5, bidirectional=False, pretrained_embeddings=None):
        super(SentimentLSTM, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
