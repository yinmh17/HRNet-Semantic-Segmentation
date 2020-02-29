import torch
from torch.autograd import Function

from .. import mat_expand_cuda


class MatExpandFunction(Function):

    @staticmethod
    def forward(ctx,
                input):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))

        ctx.save_for_backward(input)

        output = input.new_empty(
            MatExpandFunction._output_size(input))

        if not input.is_cuda:
            raise NotImplementedError
        else:
            mat_expand_cuda.mat_expand_forward_cuda(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]

        grad_input = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                mat_expand_cuda.mat_expand_backward_cuda(
                    input, grad_output, grad_input)

        return grad_input

    @staticmethod
    def _output_size(input):
        height = (input.size(2)+1)//2
        width = (input.size(3)+1)//2
        output_size = (input.size(0), input.size(1), height*width, height*width)
        if not ((2*height-1) == input.size(2) and (2*width-1) == input.size(3)):
            raise ValueError(
                "input size: {}x{} not valid".format(input.size(2), input.size(3)))
        return output_size

mat_expand = MatExpandFunction.apply
