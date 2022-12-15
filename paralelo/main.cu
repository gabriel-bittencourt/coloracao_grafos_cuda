#include <math.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <iostream>
#include <vector>
#include "helpers.h"
#include "matrix_file.cu"

#define MAX 2
#define MIN 1

using namespace std;

__global__ void min_max(int* d_adj_list,
                        int* d_degree,
                        int* d_weights,
                        int* d_min_max,
                        int N,
                        int M) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int v = index;

  // Thread não possui vértice
  if (v >= N)
    return;

  int v_weight = d_weights[v], w_weight;

  if (v_weight == -1)
    return;

  bool is_v_max = true;
  bool is_v_min = true;

  // Percorre todos as arestas do vértice v
  for (int i = 0; i < d_degree[v]; i++) {
    int w = d_adj_list[v * M + i];
    w_weight = d_weights[w];
    if (w_weight == -1)
      continue;

    if (v_weight < w_weight)
      is_v_max = false;

    else if (v_weight > w_weight)
      is_v_min = false;
  }

  // v não tem vizinhos válidos
  if (is_v_max && is_v_min)
    d_min_max[v] = MIN;

  else if (is_v_min)
    d_min_max[v] = MIN;
  else if (is_v_max)
    d_min_max[v] = MAX;
}

__global__ void color_vertices(int* d_weights,
                               int* d_min_max,
                               int* d_coloring,
                               int* current_color,
                               int* n_colored_vertices,
                               int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int v = index;

  // Thread não possui vértice
  if (v >= N)
    return;

  // Vértice não deve ser colorido nessa iteração
  if (!d_min_max[v])
    return;

  d_coloring[v] = *current_color + d_min_max[v];
  d_weights[v] = -1;
}

__global__ void update_control(int* d_min_max,
                               int* d_coloring,
                               int* current_color,
                               int* n_colored_vertices,
                               int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int v = index;

  // Thread não possui vértice
  if (v >= N)
    return;

  if (!d_min_max[v])
    return;

  // Atualiza variáveis de controle
  atomicMax(current_color, d_coloring[v]);
  atomicAdd(n_colored_vertices, 1);

  // reseta o vetor de min e max
  d_min_max[v] = 0;
}

int main(int argc, char const* argv[]) {
  // ler argumentos da linha de comando
  if (argc <= 1) {
    printf("Especifique o arquivo de entrada!\n");
    return EXIT_FAILURE;
  }
  char const* matrix_file_name = argv[1];

  // Grafo na estrutura de lista de adjacência
  printf("Lendo arquivo...\n");
  vector<vi> graph = read_matrix(matrix_file_name);
  printf("Completo!\n");

  int M = INT_MIN;  // maior grau
  int N = graph.size();
  size_t size = sizeof(int);
  int g_n_colored_vertices, n_colors_used;

  srand(time(NULL));

  int degree[N], keys[N];

  for (int i = 0; i < N; i++) {
    keys[i] = rand();
    degree[i] = graph[i].size();  // grau do vértice
    M = max((int)M, degree[i]);   // maior grau
  }

  printf("M = %d | N = %d\n", M, N);

  // Vetor de coloração (host)
  int* g_coloring = (int*)malloc(size * N);

  int *d_n_colored_vertices,  // Número de vértices coloridos
      *d_current_color;       // Cor atual

  int* d_adj_list;  // lista de adjacência

  int *d_degree,    // Número de vizinhos dos vértices
      *d_weights,   // Vetor de pesos no device
      *d_min_max,   // Vetor de minmax (device)
      *d_coloring;  // Vetor com coloração (device)

  // Aloca variável no device e inicializa com 0
  cudaMalloc((void**)&d_n_colored_vertices, size);
  cudaMemset(d_n_colored_vertices, 0, size);

  // Aloca variável no device e inicializa com -1
  cudaMalloc((void**)&d_current_color, size);
  cudaMemset(d_current_color, -1, size);

  // Aloca grafo no device e copia para lá
  cudaMalloc((void**)&d_adj_list, N * M * sizeof(int*));
  for (int i = 0; i < N; i++) {
    cudaMemcpy(d_adj_list + (i * M), &graph[i][0], size * graph[i].size(),
               cudaMemcpyHostToDevice);
  }

  // Aloca vetor com tamanho das vizinhanças
  cudaMalloc((void**)&d_degree, size * N);
  cudaMemcpy(d_degree, degree, N * size, cudaMemcpyHostToDevice);

  // Aleatorização dos pesos:
  cudaMalloc((void**)&d_weights, N * size);  // aloca o vetor de pesos
  // transforma em objeto thrust
  thrust::device_ptr<int> t_weights = thrust::device_pointer_cast(d_weights);
  // inicializa como uma sequencia
  thrust::sequence(t_weights, t_weights + N, 0);
  // vetor das keys aleatórias no device
  thrust::device_vector<int> d_keys(keys, keys + N);
  // ordena os pesos de acordo com as keys
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), t_weights);

  // Aloca vetor de minmax no device e inicializa com 0s
  cudaMalloc((void**)&d_min_max, N * size);
  cudaMemset(d_min_max, 0, N * size);

  // Aloca vetor de coloração no device e inicializa com 0s
  cudaMalloc((void**)&d_coloring, N * size);

  // Número de blocos e threads por bloco
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int n_threads = props.maxThreadsPerBlock;
  int n_blocks = ceil((double)N / (double)n_threads);

  do {
    min_max<<<n_blocks, n_threads>>>(d_adj_list, d_degree, d_weights, d_min_max,
                                     N, M);
    cudaDeviceSynchronize();
    color_vertices<<<n_blocks, n_threads>>>(d_weights, d_min_max, d_coloring,
                                            d_current_color,
                                            d_n_colored_vertices, N);
    cudaDeviceSynchronize();

    update_control<<<n_blocks, n_threads>>>(
        d_min_max, d_coloring, d_current_color, d_n_colored_vertices, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&g_n_colored_vertices, d_n_colored_vertices, size,
               cudaMemcpyDeviceToHost);

  } while (g_n_colored_vertices < N);

  // Aguarda todos os threads terminarem
  cudaDeviceSynchronize();

  cudaMemcpy(&n_colors_used, d_current_color, size, cudaMemcpyDeviceToHost);
  printf("%d cores usadas.\n", n_colors_used);

  cudaMemcpy(g_coloring, d_coloring, N * size, cudaMemcpyDeviceToHost);

  // Imprime a coloração
  printf("Coloração: ");
  for (int i = 0; i < N; i++) {
    printf("%d ", g_coloring[i]);
  }
  printf("\n");

  // Libera memória do device
  cudaFree(d_adj_list);
  cudaFree(d_degree);
  cudaFree(d_weights);
  cudaFree(d_min_max);
  cudaFree(d_coloring);
  cudaFree(d_current_color);
  cudaFree(d_n_colored_vertices);
  // // Libera memória do host
  // free(g_coloring);

  // free(weights);
  // for(int i = 0; i < N; i++)
  //     free(graph[i]);
  // free(graph);

  return 0;
}
