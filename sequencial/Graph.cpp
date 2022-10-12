#include "Graph.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using namespace std;

Graph::Graph(int V) {
    /**
     * @brief Construct a new Graph object
     * 
     * @param V Number of vertices
     */
    this->V = V;
    this->E = 0;
    this->adjacencyMatrix = vector<vector<bool>>(V, vector<bool>(V, false));

}

Graph::Graph() {
    this->V = 0;
    this->E = 0;
    this->adjacencyMatrix = vector<vector<bool>>();
}

Graph::~Graph(){
    this->adjacencyMatrix.clear();
}

void Graph::printGraph() {
    /**
     * @brief Prints the graph in the console.
     */
    for (int i = 0; i < V; i++) {
        std::cout << "| " << i << " | ";

        for (int j = 0; j < V; j++) {
            if (adjacencyMatrix[i][j] != 0) {
                std::cout << j << " ";
            }
        }
        std::cout << endl;
    }
    std::cout << endl;
}

void Graph::printAdjacencyMatrix() {
    /**
     * @brief Prints the adjacency matrix
     */
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            std::cout << adjacencyMatrix[i][j] << " ";
        }
        std::cout << endl;
    }
    std::cout << endl;
}

void Graph::addEdge(int v, int w) {
    /**
     * @brief Add an edge between vertices v and w.
     *
     * @param v The first vertex.
     * @param w The second vertex.
     */
    adjacencyMatrix[v][w] = true;
    adjacencyMatrix[w][v] = true;
    E++;
}

void Graph::removeEdge(int v, int w) {
    /**
     * @brief Remove an edge between vertices v and w.
     *
     * @param v The first vertex.
     * @param w The second vertex.
     */
    adjacencyMatrix[v][w] = false;
    adjacencyMatrix[w][v] = false;
    E--;
}


int Graph::getVertexDegree(int v) {
    /**
     * @brief Get the degree of a vertex.
     *
     * @param v The vertex.
     * @return The degree of the vertex.
     */
    int count = 0;
    for (int i = 0; i < V; i++)
        if (adjacencyMatrix[v][i] != 0)
            count++;
    return count;
}


int Graph::getV() {
    return V;
}

int Graph::getE() {
    return E;
}

vector<vector<bool>> Graph::getAdjacencyMatrix() {
    return adjacencyMatrix;
}
