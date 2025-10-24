import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, size_average="mean"):
        super(FocalLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

        assert size_average in ["sum", "mean", "none"], "Supported size_average is : sum, mean, none."
        self.size_average = size_average

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """

        Args:
            inputs: (C), (N,C), (N, C, d1, d2...dk)
            targets: (), (N, ), (N, d1, d2...dk)

        Returns:

        """
        if inputs.ndim == 1 and targets.ndim == 0:
            inputs = torch.unsqueeze_copy(inputs, dim=0)
            targets = torch.unsqueeze_copy(targets, dim=0)

        if inputs.shape[0:1] + inputs.shape[2:] != targets.shape:
            raise RuntimeError('''Inputs Shape (C), (N, C), (N, C, d_1, d_2, ..., d_K) must 
                             be corresponding to Targets Shape (), (N, ), (N, d_1, d_2, ..., d_K)''')

        n_classes = inputs.shape[1]
        if self.weight is None:
            weight = torch.tensor([1.0] * n_classes).to(inputs.device)
        else:
            if self.weight.ndim != 1:
                raise RuntimeError("weight must be None or 1-dim tensor")

            if self.weight.shape[0] != n_classes:
                raise RuntimeError("The length of weight != the number of classes of inputs.")

            weight = self.weight.to(inputs.device)

        targets = torch.unsqueeze_copy(targets, dim=1)

        input_log_softmax = F.log_softmax(inputs, dim=1)
        input_softmax = torch.exp(input_log_softmax)

        p_softmax = input_softmax.gather(dim=1, index=targets)
        p_log_softmax = input_log_softmax.gather(dim=1, index=targets)
        alpha = weight.gather(0, targets.view(-1)).view(targets.shape)

        pt = -alpha * (1 - p_softmax) ** self.gamma
        loss = pt * p_log_softmax

        if self.size_average == "mean":
            v = loss.sum() / alpha.sum()
        elif self.size_average == "sum":
            v = loss.sum()
        else:
            v = loss.squeeze(dim=1)

        return v


if __name__ == "__main__":
    import random

    n_class = 3
    class_weight = torch.rand(n_class)
    input_shape = (5, 3)

    loss_1a = FocalLoss(gamma=2, weight=class_weight, size_average="none")
    loss_1b = FocalLoss(gamma=2, weight=class_weight, size_average="mean")
    loss_1c = FocalLoss(gamma=2, weight=class_weight, size_average="sum")

    loss_2a = nn.CrossEntropyLoss(weight=class_weight, reduction="none")
    loss_2b = nn.CrossEntropyLoss(weight=class_weight, reduction="mean")
    loss_2c = nn.CrossEntropyLoss(weight=class_weight, reduction="sum")

    x = torch.randn(input_shape)
    y = torch.tensor([random.sample(range(n_class), 1)[0]
                      for _ in range(int(x.shape.numel() / n_class))]).view(x.shape[0:1] + x.shape[2:])

    v1a = loss_1a(x, y)
    v1b = loss_1b(x, y)
    v1c = loss_1c(x, y)

    v2a = loss_2a(x, y)
    v2b = loss_2b(x, y)
    v2c = loss_2c(x, y)


    print(v1a,v1b,v1c,v2a,v2b,v2c)