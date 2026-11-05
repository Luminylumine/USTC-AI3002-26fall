"""需补全：MLP 的模型构造与前向传播。"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, widths, activation="relu"):
        super().__init__()
        sizes = [input_size, *widths]

        # hidden：按顺序保存带偏置的隐藏层 Linear，类型为 nn.ModuleList。
        # activation：按参数选择 nn.ReLU() 或 nn.Sigmoid()。
        # output：从最后一个隐藏层映射到 1；无隐藏层时从 input_size 映射到 1。
        self.hidden = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1], bias=True)
            for i in range(len(widths))
        ])
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        out_in = widths[-1] if len(widths) > 0 else input_size
        self.output = nn.Linear(out_in, 1, bias=True)

    def forward(self, x):
        # 输入形状为 (B, input_size)，返回形状为 (B, 1)。
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output(x)
