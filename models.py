import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class sentiment_analysis(nn.Module):
    def __init__(self, args):
        super(sentiment_analysis, self).__init__()

        self.lstm = nn.LSTM(input_size=300, hidden_size=args.hidden_size, 
                            num_layers=1, bias=True)
        self.full = nn.Linear(in_features=args.hidden_size, out_features=2)
        self.out = nn.Softmax()

    def forward(self, rw_vec, lengths):

        

        orig_len = rw_vec.size(1)
        lengths, sort_index = lengths.sort(0, descending=True)
        rw_vec = Variable(rw_vec[sort_index])
        print(lengths)
        print(rw_vec)
        rw_vec = pack_padded_sequence(input=rw_vec, lengths=lengths, batch_first=True)
        print(rw_vec)
        lstm_out, (h_n, c_n) = self.lstm(rw_vec)
        print(f"lstm_out dim: {lstm_out.size()}")
        print(f"h_n dim: {h_n.size()}")
        print(f"c_n dim: {c_n.size()}")
#        f_out = self.full(lstm_out)
#        out = self.out(f_out)

        return None


