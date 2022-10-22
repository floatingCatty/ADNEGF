# import torch
#
# class fn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, a):
#         ctx.save_for_backward(a)
#         return x*a
#
#     def backward(ctx, *grad_outputs):
#         a = ctx.saved_tensors[0]
#
#
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import torch
    from torch.autograd.functional import hessian

    a = torch.randn(5,5)
    a[:1,:1] = 0
    print(a.bool())
    print(a)