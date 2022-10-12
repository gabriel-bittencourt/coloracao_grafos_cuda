#ifndef COLORING_H
#define COLORING_H

#include <stack>
#include <vector>

#include "Graph.h"


typedef struct Solution {
    vector<int> colors;
    int n_colors;

    Solution(int size);
    void destroy();
} Solution;


class Coloring {

    private:
        Graph *graph;
        Solution *solution;

    public:
        Coloring(Graph *graph);
        ~Coloring();

        vector<int> generate_random_weights();
        vector<int> min_max(vector<int> weights);

        Solution *greedy_coloring();
};


#endif  // COLORING_H
