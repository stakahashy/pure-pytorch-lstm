# pure-pytorch-lstm
An implementation of LSTM purely written in PyTorch.
It at least works under PyTorch=0.4.1.

## Description
The code is intended to reproduce the result of [torch.nn.LSTM](https://pytorch.org/docs/stable/nn.html#lstm).
The speed is much slower than torch.nn.LSTM as it uses cuDNNLSTM, whose implementations that "may
be many times faster due to low-level hardware-specific optimizations."

## Implementaiton Difference with torch.nn.LSTM
1. Bidirectional option is not supported
2. Weight drop to hidden to hidden weight is supported via wdrop option
