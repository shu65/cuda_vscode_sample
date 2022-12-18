#include <cassert>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void vecAdd(float *a, float *b, float *c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

int main()
{
  const int n_blocks = 128;
  const int n_threads = 512;
  const int n = n_blocks * n_threads;

  vector<float> h_a(n);
  vector<float> h_b(n);
  vector<float> h_c(n);

  for (int i = 0; i < n; ++i)
  {
    h_a[i] = i;
    h_b[i] = 2 * i;
    h_c[i] = 0;
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;

  size_t array_size = sizeof(float) * n;

  checkCudaErrors(cudaMalloc(&d_a, array_size));
  checkCudaErrors(cudaMalloc(&d_b, array_size));
  checkCudaErrors(cudaMalloc(&d_c, array_size));

  checkCudaErrors(cudaMemcpy(d_a, h_a.data(), array_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(d_b, h_b.data(), array_size, cudaMemcpyDefault));

  vecAdd<<<n_blocks, n_threads>>>(d_a, d_b, d_c);

  checkCudaErrors(cudaMemcpy(h_c.data(), d_c, array_size, cudaMemcpyDefault));

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  d_a = nullptr;
  d_b = nullptr;
  d_c = nullptr;

  for (int i = 0; i < n; ++i)
  {
    assert(h_c[i] == (h_a[i] + h_b[i]));
  }
  cout << "OK!" << endl;

  return 0;
}