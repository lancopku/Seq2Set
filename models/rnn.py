import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)
        self.config = config

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if not self.config.bidirec:
            return outputs, (h, c)
        else:
            batch_size = h.size(1)
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            state = (h.transpose(0, 1), c.transpose(0, 1))
            return outputs, state


class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.gated = nn.Sequential(nn.Linear(config.encoder_hidden_size, 1), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        p = self.gated(outputs)
        outputs = outputs * p
        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.decoder_hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.linear_v = nn.Linear(config.decoder_hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        self.hidden_size = config.decoder_hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, inputs, init_state, contexts):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        for emb in embs.split(1):
            output, state = self.rnn(emb.squeeze(0), state)
            output, attn_weights = self.attention(output, contexts)
            output = self.dropout(output)
            outputs += [output]
            attns += [attn_weights]
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)
        #outputs = self.linear(outputs.view(-1, self.hidden_size))
        return outputs, state

    def compute_score(self, hiddens):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
        else:
            scores = self.linear(hiddens)
        return scores

    def sample(self, input, init_state, contexts):
        #emb = self.embedding(input)
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        for i in range(max_time_step):
            # output: [batch, tgt_vocab_size]
            output, state, attn_weights = self.sample_one(inputs[i], state, contexts)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, state, contexts):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        hidden, attn_weigths = self.attention(output, contexts)
        output = self.compute_score(hidden)

        return output, state, attn_weigths
