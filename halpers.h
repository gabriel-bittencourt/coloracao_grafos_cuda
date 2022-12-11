// macros para checar erros sem sujar o código

#define cudaMalloc(A, B)    \
  do {                      \
    cudaError_t err;        \
    err = cudaMalloc(A, B); \
    __checkError(err);      \
  } while (0)

#define cudaMemset(A, B, C)    \
  do {                         \
    cudaError_t err;           \
    err = cudaMemset(A, B, C); \
    __checkError(err);         \
  } while (0)

#define cudaMemcpy(A, B, C, D)    \
  do {                            \
    cudaError_t err;              \
    err = cudaMemcpy(A, B, C, D); \
    __checkError(err);            \
  } while (0)

#define cudaFree(A)    \
  do {                 \
    cudaError_t err;   \
    err = cudaFree(A); \
    __checkError(err); \
  } while (0)

#define __checkError(E)                                  \
  do {                                                   \
    if (E != cudaSuccess) {                              \
      printf("CUDA error: %s\n", cudaGetErrorString(E)); \
      return E;                                          \
    }                                                    \
  } while (0)
