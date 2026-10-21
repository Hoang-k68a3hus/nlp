import torch
import torch.nn as nn


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.Wxh = torch.randn(input_size, hidden_size) * 0.1
        self.Whh = torch.randn(hidden_size, hidden_size) * 0.1
        self.bh = torch.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        a_t = x_t @ self.Wxh + h_prev @ self.Whh + self.bh
        h_t = torch.tanh(a_t)
        return h_t


if __name__ == "__main__":
    cell = SimpleRNNCell(2, 2)
    cell.Wxh = torch.tensor([[0.5, 0.1], [0.2, 0.4]])
    cell.Whh = torch.tensor([[0.3, 0.2], [0.1, 0.5]])

    x_t = torch.tensor([1.0, 0.0])
    h_prev = torch.tensor([0.5, 0.2])

    a_t = x_t @ cell.Wxh + h_prev @ cell.Whh + cell.bh
    h_t = cell.forward(x_t, h_prev)

    rnn = nn.RNNCell(2, 2)
    with torch.no_grad():
        rnn.weight_ih.copy_(cell.Wxh.T)
        rnn.weight_hh.copy_(cell.Whh.T)
        rnn.bias_ih.copy_(cell.bh)
        rnn.bias_hh.zero_()

    h_torch = rnn(x_t, h_prev)

    print("a_t:", a_t)
    print("h_t:", h_t)
    print("nn.RNNCell:", h_torch)
    print("Kết quả giống nhau:", torch.allclose(h_t, h_torch, atol=1e-6))

    assert torch.allclose(h_t, h_torch, atol=1e-6)
