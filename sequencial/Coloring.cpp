#include <bits/stdc++.h>
#include <iostream>
#include <vector>

#include "Coloring.h"
#include "Graph.h"


using namespace std;


Solution::Solution(int size) {
    this->colors = vector<int>(size, 0);
    this->n_colors = 0;
}


Coloring::Coloring(Graph *graph) {
    this->graph = graph;
    this->solution = new Solution(graph->getV());
}

Coloring::~Coloring() {
    delete this->solution;
}


vector<int> Coloring::generate_random_weights() {
    vector<int> weights(this->graph->getV());
    
    std::iota(weights.begin(), weights.end(), 0);      // Fill with 0, 1, ..., n-1
    std::mt19937 rng(std::random_device{}());          // Random-number engine
    std::shuffle(weights.begin(), weights.end(), rng); // Shuffle the array

    return weights;
}


vector<int> Coloring::min_max(vector<int> weights){
    /**
     * @brief Returns a vector where the value at the i-th position indicates
     *        whether the i-th vertex is a local min, max or neither.
     *        0 - Neither
     *        1 - Local min
     *        2 - Local max
     * 
     * @param weights Vector of weights
     * @return vector<int> Vector of local min, max and neither
     */
    vector<int> min_max(this->graph->getV(), 0);

    vector<vector<bool>> adj = this->graph->getAdjacencyMatrix();

    int v_weight, w_weight;
    bool is_v_max, is_v_min;

    for (int i = 0; i < this->graph->getV(); i++) {
        
        if (weights[i] == -1)
            continue;

        is_v_max = true, is_v_min = true;

        v_weight = weights[i];

        for (int j = 0; j < this->graph->getV(); j++) {

            if (adj[i][j]) {

                w_weight = weights[j];

                if (v_weight < w_weight)
                    is_v_max = false;

                else if (v_weight > w_weight)
                    is_v_min = false;
            }
        }

        if (is_v_min)
            min_max[i] = 1; // v is a min
        
        else if (is_v_max)
            min_max[i] = 2; // v is a max

    }

    return min_max;
}


Solution *Coloring::greedy_coloring(){
    /**
     * @brief Greedy coloring algorithm
     * 
     * @return Solution* Solution to the coloring problem
     */
    vector<int> weights = this->generate_random_weights();
    vector<int> min_max;

    vector<vector<bool>> adj = this->graph->getAdjacencyMatrix();

    Solution *solution = new Solution(this->graph->getV());

    int current_color = 1;
    int max_color = this->graph->getV() + 1;

    int n_colored_vertices = 0;

    do {
        min_max = this->min_max(weights);

        for (int i = 0; i < this->graph->getV(); i++) {

            if ( !min_max[i] )
                continue;

            // The vertex is a local min
            if (min_max[i] == 1){
                solution->colors[i] = current_color;
            }

            // The vertex is a local max
            else if (min_max[i] == 2) {
                solution->colors[i] = current_color + 1;
            }

            weights[i] = -1;      // Mark as colored
            n_colored_vertices++; // Increment the number of colored vertices
        }

        current_color += 2; // Update the current color

    } while(n_colored_vertices < this->graph->getV());

    solution->n_colors = current_color + 1;
    return solution;
}
