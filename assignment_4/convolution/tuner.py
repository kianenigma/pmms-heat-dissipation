import numpy
from kernel_tuner import run_kernel, tune_kernel

kernel_string = """
__global__ void convolution_kernel(float *output, float *input, float *filter) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
    float sum = 0.0;

    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
        }
    }
    output[y*image_width+x] = sum; 
}
"""

filter_size = (5,5)
output_size = (2048, 2048)

size = numpy.prod(output_size)
border_size = (filter_size[0]//2*2, filter_size[1]//2*2)
input_size = ((output_size[0]+border_size[0]) * (output_size[1]+border_size[1]))

output_image = numpy.zeros(size).astype(numpy.float32)
input_image = numpy.random.randn(input_size).astype(numpy.float32)

conv_filter = numpy.random.randn(filter_size[0]*filter_size[1]).astype(numpy.float32)

kernel_name = "convolution_kernel"
problem_size = output_size
arguments = [output_image, input_image, conv_filter]
params = dict()
params["block_size_x"]  = [16, 32, 64, 128]
params["block_size_y"]  = [8, 16]
params["image_height"]  = [output_size[1]]
params["image_width"]   = [output_size[0]]
params["filter_height"] = [filter_size[1]]
params["filter_width"]  = [filter_size[0]]
params["input_width"]   = [output_size[0] + border_size[0]]

results, env = tune_kernel(kernel_name, kernel_string, problem_size, arguments, params, verbose=True)

# print(results, env)