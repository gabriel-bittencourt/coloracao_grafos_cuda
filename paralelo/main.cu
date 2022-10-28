#include <iostream>
#include <stdio.h>


__global__ void min_max(int *d_graph, int *d_weights, int *d_min_max, int N){

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int v = index;
    int v_weight = d_weights[v], w_weight;

    if (v_weight == -1) return;

    bool is_v_max = true;
    bool is_v_min = true;

    // Percorre todos as arestas do vértice v
    for(int w = 0; w < N; w++){

        if (d_weights[w] == -1) continue;

        // Recupera a aresta entre v e w no grafo
        int edge = d_graph[v * N + w];
        
        // Se existe aresta entre v e w
        if (edge){
            w_weight = d_weights[w];

            if (v_weight < w_weight)
                is_v_max = false;

            else if (v_weight > w_weight)
                is_v_min = false;
            
        }
    }

    // v não tem vizinhos válidos
    if (is_v_max && is_v_min)
        d_min_max[v] = 1;

    else if (is_v_min) d_min_max[v] = 1;
    else if (is_v_max) d_min_max[v] = 2;

}

int main() {

    // Grafo na estrutura de matriz de adjacência
    // TODO: ler do arquivo
    int N = 10;
    int graph[10][10] = {
    //   0  1  2  3  4  5  6  7  8  9
        {0, 1, 0, 0, 1, 1, 0, 0, 0, 0}, // 0
        {1, 0, 1, 0, 0, 0, 1, 0, 0, 0}, // 1
        {0, 1, 0, 1, 0, 0, 0, 1, 0, 0}, // 2
        {0, 0, 1, 0, 1, 0, 0, 0, 1, 0}, // 3
        {1, 0, 0, 1, 0, 0, 0, 0, 0, 1}, // 4
        {1, 0, 0, 0, 0, 0, 1, 0, 0, 0}, // 5
        {0, 1, 0, 0, 0, 1, 0, 1, 0, 0}, // 6
        {0, 0, 1, 0, 0, 0, 1, 0, 1, 0}, // 7
        {0, 0, 0, 1, 0, 0, 0, 1, 0, 1}, // 8
        {0, 0, 0, 0, 1, 1, 0, 0, 1, 0}  // 9
    };

    size_t size = sizeof(int);

    // Vetor de pesos
    // TODO: Gerar aleatoriamente
    int weights[] = {9, 4, 3, 2, 8, 7, 6, 1, 0, 5};

    // Vetor com coloração (host)
    int *coloring = (int*) malloc(size * N);

    // Vetor de minmax (host)
    int *g_min_max = (int*) malloc(size * N);

    int *d_graph,   // Grafo no device (array 1D)
        *d_weights, // Vetor de pesos no device
        *d_min_max; // Vetor de minmax (device)

    // Aloca grafo no device e copia para lá
    cudaMalloc((void**) &d_graph, N * N * size);
    cudaMemcpy(d_graph, graph, N * N * size, cudaMemcpyHostToDevice);
    
    // Aloca vetor de pesos no device e copia para lá
    cudaMalloc((void**) &d_weights, N * size);
    cudaMemcpy(d_weights, weights, N * size, cudaMemcpyHostToDevice);
    
    // Aloca vetor de minmax no device e inicializa com 0s
    cudaMalloc((void**) &d_min_max, N * size);
    cudaMemset(d_min_max, 0, N * size);

    // Número de blocos e threads por bloco
    // TODO: Usar cudaDeviceProp
    int n_blocks = 1;
    int n_threads = N;

    int n_colored_vertices = 0;      // Número de vértices coloridos
    int current_color = 0;           // Cor atual
    bool used_second_color = false;  // Usou a segunda cor

    do {
        // Calcula os vértices que são mínimos e máximos
        min_max<<<n_blocks, n_threads>>>(d_graph, d_weights, d_min_max, N);
        
        // Aguarda todos os threads terminarem
        cudaDeviceSynchronize();

        // Carrega o vetor com os vértices que são min ou max para o host
        cudaMemcpy(g_min_max, d_min_max, N * size, cudaMemcpyDeviceToHost);

        for(int i = 0; i < N; i++){
            
            // Se o vértice i não é nem máximo nem mínimo
            if (!g_min_max[i]) continue;

            // Se o vértice i é máximo
            if (g_min_max[i] == 1)
                coloring[i] = current_color;

            // Se o vértice i é mínimo
            else if (g_min_max[i] == 2){
                coloring[i] = current_color + 1;
                used_second_color = true;
            }

            weights[i] = -1;
            n_colored_vertices++;

        }

        // Atualiza o vetor de pesos no device
        // TODO: Atualizar esses pesos só na memória do device
        cudaMemcpy(d_weights, weights, N * size, cudaMemcpyHostToDevice);

        // Atualiza a cor atual
        if (used_second_color) current_color += 2;
        else current_color++;

        used_second_color = false;

        // Reseta o vetor de min e max
        // TODO (?): Resetar esse vetor só na memória do device
        cudaMemset(d_min_max, 0, N * size);

    } while (n_colored_vertices < N); // Enquanto não colorir todos os vértices

    // Imprime a coloração
    printf("Coloração: ");
    for (int i = 0; i < N; i++){
        printf("%d ", coloring[i]);
    }
    printf("\n");

    // Libera memória do device
    cudaFree(d_graph);
    cudaFree(d_weights);
    cudaFree(d_min_max);

    // Libera memória do host
    free(coloring);
    free(g_min_max);
    // free(weights);
    // for(int i = 0; i < N; i++)
    //     free(graph[i]);
    // free(graph);

    return 0;
}
