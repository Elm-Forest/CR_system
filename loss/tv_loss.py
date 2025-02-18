import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        """
        计算 Total Variation Loss.
        :param x: 输入图像，大小为 (N, C, H, W)
        :return: TV Loss 值
        """
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)


# 示例用法
if __name__ == "__main__":
    # 假设输入图像大小为 (1, 3, 256, 256)
    img = torch.randn(1, 3, 256, 256, requires_grad=True)
    tv_loss = TVLoss(weight=1.0)

    loss = tv_loss(img)
    print(f"TV Loss: {loss.item()}")

    # 假设这是你的优化器
    optimizer = torch.optim.Adam([img], lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
