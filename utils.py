import torch


# 가중치 초기화
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def set_requires_grad(model, requires_grad=False):
    if not isinstance(model, list):
        model = [model]
    for _model in model:
        if _model is not None:
            for param in _model.parameters():
                param.requires_grad = requires_grad