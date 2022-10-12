#include <iostream>

#include "Coloring.h"
#include "Graph.h"

using namespace std;


int main() {
    
    int n = 10;
    Graph *graph = new Graph(n);

    graph->addEdge(0, 1);
    graph->addEdge(1, 2);
    graph->addEdge(2, 3);
    graph->addEdge(3, 4);
    graph->addEdge(4, 0);

    graph->addEdge(0, 5);
    graph->addEdge(1, 6);
    graph->addEdge(2, 7);
    graph->addEdge(3, 8);
    graph->addEdge(4, 9);

    graph->addEdge(5, 6);
    graph->addEdge(6, 7);
    graph->addEdge(7, 8);
    graph->addEdge(8, 9);
    graph->addEdge(9, 5);

    Coloring *coloring = new Coloring(graph);

    Solution *solution = coloring->greedy_coloring();

    for(int i = 0; i < n; i++) {
        cout << solution->colors[i] << " ";
    }
    cout << endl;

    return 0;
}
