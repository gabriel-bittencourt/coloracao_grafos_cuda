#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

typedef vector<int> vi;

vector<vi> read_matrix(string filename) {
  ifstream input_file(filename.c_str(), ios_base::in);

  if (!input_file.is_open()) {
    cerr << "Could not open the file - '" << filename << "'" << endl;
    return vector<vi>();
  }

  // ler dimensões da matriz e do arquivo
  int N, L;
  input_file >> N >> N >> L;
  vector<vi> graph(N);

  int i, j, count = 1;
  float k;

  while (input_file >> i >> j >> k) {
    i--, j--;  // 1 based
    graph[i].push_back(j);
    graph[j].push_back(i);
    count++;
  }

  input_file.close();

  cout << count << " entradas obtidas." << endl;
  return graph;
}
