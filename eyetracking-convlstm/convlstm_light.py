import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor):
        # h_cur, c_cur, h_pre = cur_state
        # delta = h_cur - h_pre
        # threshold = torch.tensor(float("inf")).cuda()
        # delta = torch.where(delta < threshold, torch.tensor(0.0).cuda(), delta)
        combined = torch.cat([input_tensor], dim=1)  # concatenate along channel axis
        # if self.training == False:
        #     non_zero_count = torch.count_nonzero(combined).float()
        #     sparse_rate = (combined.numel() - non_zero_count) / combined.numel()
        #     sparse_rate_inp = (input_tensor.numel() - torch.count_nonzero(input_tensor).float()) / input_tensor.numel()
        #     sparse_rate_delta = (delta.numel() - torch.count_nonzero(delta).float()) / delta.numel()
        #     file_path = os.path.join(log_dir, f"sparse_rate_th_{threshold:.5f}.txt")
        #     # file_path = os.path.join(log_dir, f"sparse_rate_base.txt")
        #     with open(file_path, 'a') as f:
        #         f.write(f"sparse_rate: tot {sparse_rate} inp {sparse_rate_inp} delta {sparse_rate_delta} size {combined.shape}\n")

            # fig, axs = plt.subplots(4, 8, figsize=(10, 10))
            # for i, ax in enumerate(axs.flatten()):
            #     ax.imshow(np.array(delta[0,i,:,:].cpu()), cmap='gray')  # Displaying it in grayscale for this example
            #     ax.axis('off')
            #
            # picname = f'hidden_{threshold:.5f}_{input_tensor.size(-1)}.png'
            # plt.savefig(os.path.join(log_dir, picname))
            # plt.close()
        combined_conv = torch.relu(self.conv(combined))

        # cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        #
        # i = torch.sigmoid(cc_i)
        # f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        # g = torch.tanh(cc_g)
        #
        # c_next = f * torch.relu(c_cur) + i * g
        # h_next = o * torch.relu(c_next)
        #
        # return h_next, c_next, h_cur

        return combined_conv

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, 4 * self.hidden_dim, height, width, device=self.conv.weight.device)


class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, input_tensor, cur_state):
        c_cur = cur_state
        cc_i, cc_f, cc_o, cc_g = torch.split(input_tensor, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * torch.relu(c_cur) + i * g
        h_next = o * torch.relu(c_next)

        return h_next, c_next


    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        lstm_cell_list =[]
        bn_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            lstm_cell_list.append(LSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i]
                                          ))
            bn_list.append(nn.BatchNorm3d(self.hidden_dim[i]*4))

        self.cell_list = nn.ModuleList(cell_list)
        self.lstm_cell_list = nn.ModuleList(lstm_cell_list)
        self.bn_list = nn.ModuleList(bn_list)



    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            hidden_state_lstm = self._init_hidden_lstm(batch_size=b,
                                             image_size=(h, w))
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            # combined = hidden_state[layer_idx]
            output_middle_inner = []
            for t in range(seq_len):
                combined= self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :])
                output_middle_inner.append(combined)

            layer_middle_output = torch.stack(output_middle_inner, dim=1)
            layer_middle_output = layer_middle_output.permute(0, 2, 1, 3, 4)
            layer_middle_output = self.bn_list[layer_idx](layer_middle_output)
            layer_middle_output = layer_middle_output.permute(0, 2, 1, 3, 4).to(device)
            c = hidden_state_lstm[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.lstm_cell_list[layer_idx](input_tensor=layer_middle_output[:, t, :, :, :],
                                                     cur_state=c)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    def _init_hidden_lstm(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.lstm_cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param