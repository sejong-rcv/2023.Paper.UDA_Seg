import torch
import torch.nn as nn
import torch.nn.functional as F

def edge_loss_bce(y_hat_log, y): #예측, 타켓
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            beta = y.mean(dim=[2, 3], keepdims=True)
        logit_1 = F.logsigmoid(y_hat_log)
        logit_0 = F.logsigmoid(-y_hat_log)
        loss = -(1 - beta) * logit_1 * y - beta * logit_0 * (1 - y)
        return loss.mean()