import torch.nn as nn
import torch.nn.functional as F
from modules.gcn.attention import Selfattention

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        # self.layer1 = Selfattention(input_shape, args.rnn_hidden_dim,args.rnn_hidden_dim)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs,  adj, hidden_state=None):
        b, a, e = inputs.size()
        inputs = inputs.view(-1,e)
        # x, _ = F.relu(self.layer1(inputs,inputs,inputs,adj))
        x = F.relu(self.fc1(inputs), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q.view(b, a, -1), h.view(b, a, -1)