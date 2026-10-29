"""Import-safe vanilla ConvLSTM model used by the SEET-public-v1 runner.

This is a faithful extraction of ``MyModel`` and ``convlstmbak.ConvLSTM`` from
the original public training code.  It intentionally does not claim to
implement the paper's change-based (CB) execution mechanism.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


KernelSize = Union[Tuple[int, int], Sequence[Tuple[int, int]]]
HiddenDim = Union[int, Sequence[int]]


class ConvLSTMCell(nn.Module):
    """One vanilla ConvLSTM cell, matching ``convlstmbak.py``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        bias: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self, input_tensor: Tensor, cur_state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # The ReLU here is present in the repository's original
        # ``convlstmbak.ConvLSTMCell`` and is therefore kept deliberately.
        combined_conv = torch.relu(self.conv(combined))
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self, batch_size: int, image_size: Tuple[int, int]
    ) -> Tuple[Tensor, Tensor]:
        height, width = image_size
        shape = (batch_size, self.hidden_dim, height, width)
        return (
            torch.zeros(shape, device=self.conv.weight.device),
            torch.zeros(shape, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    """Vanilla multi-layer ConvLSTM copied from the public implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: HiddenDim,
        kernel_size: KernelSize,
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()
        kernel_sizes = self._extend_kernel_size(kernel_size, num_layers)
        hidden_dims = self._extend_hidden_dim(hidden_dim, num_layers)
        if not len(kernel_sizes) == len(hidden_dims) == num_layers:
            raise ValueError("Inconsistent ConvLSTM layer configuration")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dims
        self.kernel_size = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cells: List[ConvLSTMCell] = []
        for layer_index in range(num_layers):
            current_input_dim = (
                input_dim if layer_index == 0 else hidden_dims[layer_index - 1]
            )
            cells.append(
                ConvLSTMCell(
                    input_dim=current_input_dim,
                    hidden_dim=hidden_dims[layer_index],
                    kernel_size=kernel_sizes[layer_index],
                    bias=bias,
                )
            )
        self.cell_list = nn.ModuleList(cells)

    def forward(
        self,
        input_tensor: Tensor,
        hidden_state: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[List[Tensor], List[Tuple[Tensor, Tensor]]]:
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, sequence_length, _, height, width = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError("Stateful ConvLSTM is not used by this model")
        states = self._init_hidden(batch_size, (height, width))

        layer_outputs: List[Tensor] = []
        last_states: List[Tuple[Tensor, Tensor]] = []
        current_input = input_tensor

        for layer_index in range(self.num_layers):
            h, c = states[layer_index]
            outputs: List[Tensor] = []
            for time_index in range(sequence_length):
                h, c = self.cell_list[layer_index](
                    current_input[:, time_index, :, :, :], (h, c)
                )
                outputs.append(h)
            layer_output = torch.stack(outputs, dim=1)
            current_input = layer_output
            layer_outputs.append(layer_output)
            last_states.append((h, c))

        if not self.return_all_layers:
            layer_outputs = layer_outputs[-1:]
            last_states = last_states[-1:]
        return layer_outputs, last_states

    def _init_hidden(
        self, batch_size: int, image_size: Tuple[int, int]
    ) -> List[Tuple[Tensor, Tensor]]:
        return [
            cell.init_hidden(batch_size, image_size) for cell in self.cell_list
        ]

    @staticmethod
    def _extend_kernel_size(
        kernel_size: KernelSize, num_layers: int
    ) -> List[Tuple[int, int]]:
        if isinstance(kernel_size, tuple):
            values = [kernel_size] * num_layers
        else:
            values = list(kernel_size)
        if not all(
            isinstance(value, tuple) and len(value) == 2 for value in values
        ):
            raise ValueError("kernel_size must be a tuple or a list of tuples")
        return values

    @staticmethod
    def _extend_hidden_dim(hidden_dim: HiddenDim, num_layers: int) -> List[int]:
        if isinstance(hidden_dim, int):
            return [hidden_dim] * num_layers
        return list(hidden_dim)


class MyModel(nn.Module):
    """The repository's vanilla four-layer ConvLSTM eye-tracking model."""

    def __init__(self, height: int = 60, width: int = 80, input_dim: int = 1):
        super().__init__()
        if (height, width) != (60, 80):
            raise ValueError(
                "The published fully connected layer expects 60x80 inputs"
            )

        self.convlstm1 = ConvLSTM(
            input_dim, 8, (3, 3), num_layers=1, batch_first=True
        )
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm2 = ConvLSTM(
            8, 16, (3, 3), num_layers=1, batch_first=True
        )
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm3 = ConvLSTM(
            16, 32, (3, 3), num_layers=1, batch_first=True
        )
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm4 = ConvLSTM(
            32, 64, (3, 3), num_layers=1, batch_first=True
        )
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.fc1 = nn.Linear(64 * 3 * 5, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    @staticmethod
    def _convlstm_block(
        x: Tensor,
        recurrent: ConvLSTM,
        batch_norm: nn.BatchNorm3d,
        pool: nn.MaxPool3d,
    ) -> Tensor:
        outputs, _ = recurrent(x)
        x = outputs[0].permute(0, 2, 1, 3, 4)
        x = batch_norm(x)
        x = F.relu(x)
        x = pool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._convlstm_block(x, self.convlstm1, self.bn1, self.pool1)
        x = self._convlstm_block(
            x.permute(0, 2, 1, 3, 4),
            self.convlstm2,
            self.bn2,
            self.pool2,
        )
        x = self._convlstm_block(
            x.permute(0, 2, 1, 3, 4),
            self.convlstm3,
            self.bn3,
            self.pool3,
        )
        x = self._convlstm_block(
            x.permute(0, 2, 1, 3, 4),
            self.convlstm4,
            self.bn4,
            self.pool4,
        )

        batch_size, _, sequence_length, _, _ = x.size()
        predictions: List[Tensor] = []
        for time_index in range(sequence_length):
            timestep = x[:, :, time_index, :, :].reshape(batch_size, -1)
            timestep = F.relu(self.fc1(timestep))
            timestep = self.drop(timestep)
            predictions.append(self.fc2(timestep))
        return torch.stack(predictions, dim=1)


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable and non-trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())
