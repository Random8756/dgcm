from .q_learner import QLearner
from .coma_learner import COMALearner
from .nq_learner import NQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["nq_learner"] = NQLearner