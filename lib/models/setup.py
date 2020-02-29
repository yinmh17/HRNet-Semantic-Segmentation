from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mat_expand',
    ext_modules=[
        CUDAExtension('mat_expand_cuda', [
            'src/mat_expand_cuda.cpp',
            'src/mat_expand_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
