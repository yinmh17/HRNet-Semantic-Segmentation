#include <ATen/ATen.h>

#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__global__ void lut2mat_gpu_kernel(const int n, const scalar_t *input, const int head, const int height, const int width, scalar_t *output)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_out_k = index % width;
    const int h_out_k = (index / width) % height;
    const int w_out_q = (index / width / height) % width;
    const int h_out_q = (index / width / height / width) % height;
    const int n_head = (index / width / height / width / height) % head;
    const int batch = index / width / height / width / height / head;

    const int w_in = w_out_k - w_out_q + width - 1;
    const int h_in = h_out_k - h_out_q + height - 1;

    const int input_width = 2 * width - 1;
    const int input_height = 2 * height - 1;

    const scalar_t *input_ptr = input + ((batch * head + n_head) * input_height + h_in) * input_width + w_in;
    scalar_t *output_ptr = output + index;
    *output_ptr = *input_ptr;
  }
}

void lut2mat(const at::Tensor input, const long batchSize, const long nHead, const long imgHeight, const long imgWidth, at::Tensor output)
{
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int num_kernels = batchSize * nHead * imgHeight * imgWidth * imgHeight * imgWidth;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "lut2mat_gpu", ([&] {
        const scalar_t *input_ = input.data<scalar_t>();
        scalar_t *output_ = output.data<scalar_t>();

        lut2mat_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, input_, nHead, imgHeight, imgWidth, output_
        );
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in lut2mat: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void mat2lut_gpu_kernel(const int n, const scalar_t *output, const int head, const int height, const int width, scalar_t *input)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_out_k = index % width;
    const int h_out_k = (index / width) % height;
    const int w_out_q = (index / width / height) % width;
    const int h_out_q = (index / width / height / width) % height;
    const int n_head = (index / width / height / width / height) % head;
    const int batch = index / width / height / width / height / head;

    const int w_in = w_out_k - w_out_q + width - 1;
    const int h_in = h_out_k - h_out_q + height - 1;

    const int input_width = 2 * width - 1;
    const int input_height = 2 * height - 1;

    scalar_t *input_ptr = input + ((batch * head + n_head) * input_height + h_in) * input_width + w_in;
    const scalar_t *output_ptr = output + index;
    atomicAdd(input_ptr, *output_ptr);
  }
}

void mat2lut(const at::Tensor output, const long batchSize, const long nHead, const long imgHeight, const long imgWidth, at::Tensor input)
{
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int num_kernels = batchSize * nHead * imgHeight * imgWidth * imgHeight * imgWidth;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.type(), "mat2lut_gpu", ([&] {
              const scalar_t *output_ = output.data<scalar_t>();
              scalar_t *input_ = input.data<scalar_t>();

              mat2lut_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                      num_kernels, output_, nHead, imgHeight, imgWidth, input_
              );
          }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in mat2lut: %s\n", cudaGetErrorString(err));
  }
}
