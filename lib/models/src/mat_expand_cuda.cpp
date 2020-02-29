#include <torch/extension.h>

#include <cmath>
#include <vector>

void lut2mat(const at::Tensor input, const long batchSize, const long nHead, const long imgHeight, const long imgWidth, at::Tensor output);

void mat2lut(const at::Tensor output, const long batchSize, const long nHead, const long imgHeight, const long imgWidth, at::Tensor input);

void shape_check(at::Tensor input, at::Tensor output) {

    long batchSize = input.size(0);
    long nHead = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long imgHeight = (inputHeight+1)/2;
    long imgWidth = (inputWidth+1)/2;

    AT_CHECK((inputHeight == 2*imgHeight-1), "invalid inputHeight: %d", inputHeight);
    AT_CHECK((inputWidth == 2*imgWidth-1), "invalid inputWidth: %d", inputWidth);

    long outputHeight = output.size(2);
    long outputWidth = output.size(3);

    AT_CHECK((output.size(0) == batchSize), "invalid batch size of output");
    AT_CHECK((output.size(1) == nHead), "invalid num head of output");

    AT_CHECK((outputHeight == imgHeight * imgWidth), "invalid outputHeight");
    AT_CHECK((outputWidth == imgHeight * imgWidth), "invalid outputWidth");

}

int mat_expand_forward_cuda(at::Tensor input, at::Tensor output) {

    input = input.contiguous();
    shape_check(input, output);

    long batchSize = input.size(0);
    long nHead = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long imgHeight = (inputHeight+1)/2;
    long imgWidth = (inputWidth+1)/2;

    lut2mat(input, batchSize, nHead, imgHeight, imgWidth, output);

    return 1;
}

int mat_expand_backward_cuda(at::Tensor input, at::Tensor gradOutput, at::Tensor gradInput) {
    shape_check(input, gradOutput);
    shape_check(gradInput, gradOutput);
    input = input.contiguous();
    gradOutput = gradOutput.contiguous();

    long batchSize = input.size(0);
    long nHead = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long imgHeight = (inputHeight+1)/2;
    long imgWidth = (inputWidth+1)/2;

    mat2lut(gradOutput, batchSize, nHead, imgHeight, imgWidth, gradInput);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mat_expand_forward_cuda", &mat_expand_forward_cuda,
    "mat_expand forward (CUDA)");
    m.def("mat_expand_backward_cuda", &mat_expand_backward_cuda,
    "mat_expand backward (CUDA)");
}
