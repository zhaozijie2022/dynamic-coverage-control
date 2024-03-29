"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

# from rlkit.policies.base import Policy
from utils import pytorch_utils as ptu
from torchkit.core import PyTorchModule
from torchkit.modules import LayerNorm
# from rlkit.torchkit.data_management.normalizer import TorchFixedNormalizer


class Mlp(PyTorchModule):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.gelu,
            output_activation=ptu.identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class RNNActor(PyTorchModule):
    def __init__(
        self,
        hidden_size,
        output_size,
        input_size,
        init_w = 3e-3,
        hidden_activation = F.relu,
        output_activation = ptu.identity,
        hidden_init = ptu.fanin_init,
        b_init_value = 0.1,
        layer_norm = False,
        layer_norm_kwargs = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_size).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        actions = F.tanh(self.fc2(h))
        return {"actions": actions, "hidden_state": h}



