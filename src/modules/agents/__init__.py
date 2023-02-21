REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .atten_rnn_agent import ATTRNNAgent
from .dgcm_agent import DGCMAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["DGCM"] = DGCMAgent
