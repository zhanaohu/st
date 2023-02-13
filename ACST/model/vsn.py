import torch

from model.grn import GatedResidualNetwork
import torch.nn as nn


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, context_size=None, is_temporal=None):
        super(VariableSelectionNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size
        self.is_temporal = is_temporal

        self.flattened_inputs = GatedResidualNetwork(
            self.output_size * self.input_size,
            self.hidden_size,
            self.output_size,
            self.dropout,
            self.context_size,
            self.is_temporal
        )

        self.transformed_inputs = nn.ModuleList([
            GatedResidualNetwork(
                self.input_size,
                self.hidden_size,
                self.hidden_size,
                self.dropout,
                self.context_size,
                self.is_temporal
            )
            for _ in range(self.output_size)
        ])

        self.softmax = nn.Softmax(dim=-1)

        pass

    def forward(self, embedding, context=None):
        sparse_weights = self.flattened_inputs(embedding, context)

        if self.is_temporal:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
            pass
        else:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(1)
            pass
        transformed_embeddings = torch.stack(
            [self.transformed_inputs[i](embedding[Ellipsis, i * self.input_size:(i + 1) * self.input_size]) for i in
             range(self.output_size)], axis=-1)

        combined = transformed_embeddings * sparse_weights
        combined = combined.sum(axis=-1)

        return combined, sparse_weights

    pass
