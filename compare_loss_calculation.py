import torch
import torch.nn as nn
torch.manual_seed(42)
# To compare the difference between these loss calculations. Both require logits number as input.


def xe_mannual(logits_num, y):
    softmax_output = nn.Softmax(dim=1)
    eps = 1e-4

    softmax_num = softmax_output(logits_num)
    criterion = nn.NLLLoss(reduction='none')
    return criterion(torch.log(softmax_num.clamp_min(eps)), y)


def xe_torch(logits_num, y):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    return loss_function(logits_num, y.long().view(-1))

if __name__ == '__main__':
    logits = torch.randn(5, 2)
    y = [1, 0, 0, 1, 0]
    y_tensor = torch.tensor(y)
    xe_mannual_calculated = xe_mannual(logits, y_tensor)
    xe_torch_calculated = xe_torch(logits, y_tensor)
