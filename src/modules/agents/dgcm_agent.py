import torch.nn as nn
import torch.nn.functional as F
from modules.dgcm.WGCC import WGCC_weight

class DGCMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DGCMAgent, self).__init__()
        self.args = args
        self.layer1 = WGCC_Attention(input_shape, args.rnn_hidden_dim,args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(2 * args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs,  adj, hidden_state=None):
        b, a, e = inputs.size()
        inputs = inputs.view(-1,e)
        x_ ,_ = self.layer1(inputs,inputs,inputs,adj)
        x = F.relu(x_)
        # x = F.relu(self.fc1(inputs), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc(h)
        return q.view(b, a, -1), h.view(b, a, -1)