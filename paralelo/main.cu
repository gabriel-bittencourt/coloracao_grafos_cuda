#include <stdio.h>
#include <iostream>
#include "helpers.h"

#define MAX 1
#define MIN 2

__device__ void color_vertices(int* d_weights,
                               int* d_min_max,
                               int* d_coloring,
                               int* current_color,
                               int* n_colored_vertices) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Vértice não deve ser colorido nessa iteração
  if (!d_min_max[index])
    return;

  d_coloring[index] = *current_color + d_min_max[index];
  d_weights[index] = -1;

  // Atualiza variáveis de controle
  // TODO: Algum jeito melhor pra decidir a cor atual?
  atomicMax(current_color, d_coloring[index]);
  atomicAdd(n_colored_vertices, 1);
}

__global__ void min_max(int* d_graph,
                        int* d_weights,
                        int* d_min_max,
                        int N,
                        int* d_coloring,
                        int* current_color,
                        int* n_colored_vertices) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int v = index;
  int v_weight = d_weights[v], w_weight;

  if (v_weight == -1)
    return;

  bool is_v_max = true;
  bool is_v_min = true;

  // Percorre todos as arestas do vértice v
  for (int w = 0; w < N; w++) {
    if (d_weights[w] == -1)
      continue;

    // Recupera a aresta entre v e w no grafo
    int edge = d_graph[v * N + w];

    // Se existe aresta entre v e w
    if (edge) {
      w_weight = d_weights[w];

      if (v_weight < w_weight)
        is_v_max = false;

      else if (v_weight > w_weight)
        is_v_min = false;
    }
  }

  // v não tem vizinhos válidos
  if (is_v_max && is_v_min)
    d_min_max[v] = MAX;

  else if (is_v_min)
    d_min_max[v] = MIN;
  else if (is_v_max)
    d_min_max[v] = MAX;

  __syncthreads();

  color_vertices(d_weights, d_min_max, d_coloring, current_color,
                 n_colored_vertices);

  __syncthreads();

  // Reseta o vetor de min e max
  d_min_max[v] = 0;
}

int main() {
  // Grafo na estrutura de matriz de adjacência
  // TODO: ler do arquivo
  int N = 10;
  int graph[10][10] = {
      //   0  1  2  3  4  5  6  7  8  9
      {0, 1, 0, 0, 1, 1, 0, 0, 0, 0},  // 0
      {1, 0, 1, 0, 0, 0, 1, 0, 0, 0},  // 1
      {0, 1, 0, 1, 0, 0, 0, 1, 0, 0},  // 2
      {0, 0, 1, 0, 1, 0, 0, 0, 1, 0},  // 3
      {1, 0, 0, 1, 0, 0, 0, 0, 0, 1},  // 4
      {1, 0, 0, 0, 0, 0, 1, 0, 0, 0},  // 5
      {0, 1, 0, 0, 0, 1, 0, 1, 0, 0},  // 6
      {0, 0, 1, 0, 0, 0, 1, 0, 1, 0},  // 7
      {0, 0, 0, 1, 0, 0, 0, 1, 0, 1},  // 8
      {0, 0, 0, 0, 1, 1, 0, 0, 1, 0}   // 9
  };

  size_t size = sizeof(int);

  // Vetor de pesos
  // TODO: Gerar aleatoriamente: sequência + shuffle
  int weights[] = {9, 4, 3, 2, 8, 7, 6, 1, 0, 5};

  // Vetor de coloração (host)
  int* g_coloring = (int*)malloc(size * N);
  int g_n_colored_vertices;

  int *d_n_colored_vertices,  // Número de vértices coloridos
      *d_current_color;       // Cor atual

  int *d_graph,     // Grafo no device (array 1D)
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
  cudaMalloc((void**)&d_graph, N * N * size);
  cudaMemcpy(d_graph, graph, N * N * size, cudaMemcpyHostToDevice);

  // Aloca vetor de pesos no device e copia para lá
  cudaMalloc((void**)&d_weights, N * size);
  cudaMemcpy(d_weights, weights, N * size, cudaMemcpyHostToDevice);

  // Aloca vetor de minmax no device e inicializa com 0s
  cudaMalloc((void**)&d_min_max, N * size);
  cudaMemset(d_min_max, 0, N * size);

  // Aloca vetor de coloração no device e inicializa com 0s
  cudaMalloc((void**)&d_coloring, N * size);

  // Número de blocos e threads por bloco
  // TODO: Usar cudaDeviceProp
  int n_blocks = 1;
  int n_threads = N;

  do {
    // Calcula os vértices que são mínimos e máximos
    min_max<<<n_blocks, n_threads>>>(d_graph, d_weights, d_min_max, N,
                                     d_coloring, d_current_color,
                                     d_n_colored_vertices);

    // Aguarda todos os threads terminarem
    cudaDeviceSynchronize();

    // Atualiza o número de vértices coloridos no host
    cudaMemcpy(&g_n_colored_vertices, d_n_colored_vertices, size,
               cudaMemcpyDeviceToHost);

  } while (g_n_colored_vertices < N);  // Enquanto não colorir todos os vértices

  cudaMemcpy(g_coloring, d_coloring, N * size, cudaMemcpyDeviceToHost);

  // Imprime a coloração
  printf("Coloração: ");
  for (int i = 0; i < N; i++) {
    printf("%d ", g_coloring[i]);
  }
  printf("\n");

  // Libera memória do device
  cudaFree(d_graph);
  cudaFree(d_weights);
  cudaFree(d_min_max);
  cudaFree(d_coloring);
  cudaFree(d_current_color);
  cudaFree(d_n_colored_vertices);

  // Libera memória do host
  free(g_coloring);

  // free(weights);
  // for(int i = 0; i < N; i++)
  //     free(graph[i]);
  // free(graph);

  return 0;
}
