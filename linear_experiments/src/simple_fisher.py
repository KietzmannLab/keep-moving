import torch


def compute_diagonal_fisher(model, layer, data, criterion, task):
    fish_info = torch.zeros_like(layer.weight)
    N = 0
    for x, y in data:
        # make sure x and y live on the same device as the layer parameters
        x = x.to(layer.weight.device)
        y = y.to(layer.weight.device)
        task = torch.tensor(task).to(layer.weight.device)
        model.zero_grad()
        yhat = model(x, task)
        loss = criterion(yhat, y)
        loss.backward()
        grads = layer.weight.grad
        grads = grads**2
        fish_info += grads
        N += len(x)
    fish_info /= N
    return fish_info
