#ifndef GRAPH_H
#define GRAPH_H

#include <vector>

using namespace std;

/**
 * @brief Graph class implementation using adjacency matrix
 */
class Graph {
   private:
    int V;                                 // Number of vertices
    int E;                                 // Number of edges
    vector<vector<bool>> adjacencyMatrix;  // Adjacency matrix

   public:
    Graph(int V);
    Graph();
    ~Graph();

    // Manipulate edges
    void addEdge(int v, int w);
    void removeEdge(int v, int w);

    // Print graph
    void printGraph();
    void printAdjacencyMatrix();

    // Auxiliar getters
    int getVertexDegree(int v);

    // Getters
    int getV();
    int getE();
    vector<vector<bool>> getAdjacencyMatrix();
};

#endif  // GRAPH_H
