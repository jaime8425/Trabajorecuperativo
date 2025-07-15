#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <omp.h>


using dist_t = long long;
const dist_t INF = std::numeric_limits<dist_t>::max() / 2;

// Facilitar la división de la representación en cadenas individuales para parsear la matriz de adyacencia.
static std::vector<std::string> split(const std::string &s, const std::string &delim) {
    std::vector<std::string> elems;
    size_t pos = 0, found;
    while ((found = s.find(delim, pos)) != std::string::npos) {
        elems.push_back(s.substr(pos, found - pos));
        pos = found + delim.length();
    }
    elems.push_back(s.substr(pos));
    return elems;
}

// Parsea la cadena JSON-like a matriz de adyacencia
std::vector<std::vector<int>> parse_matrix(const std::string &input) {
    // Quitar espacios
    std::string s;
    for (char c : input) if (!isspace(c)) s.push_back(c);

    // Debe empezar con "[[" y terminar con "]]"
    if (s.size() < 4 || s.rfind("[[", 0) != 0 || s.find("]]") != s.size() - 2) {
        return {};
    }
    // Extraer contenido interno
    std::string core = s.substr(2, s.size() - 4);
    // Dividir filas por "],["
    auto rows_str = split(core, "],[");

    std::vector<std::vector<int>> graph;
    for (auto &row_str : rows_str) {
        std::vector<int> row;
        std::stringstream ss(row_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                row.push_back(std::stoi(token));
            } catch (...) {
                // ignorar
            }
        }
        if (!row.empty()) graph.push_back(row);
    }
    return graph;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: programa \"[matriz]\" [vertice_origen] \"archivo_salida\"\n";
        return 1;
    }

    std::string matrix_str = argv[1];
    int src = std::stoi(argv[2]);
    std::string out_path = argv[3];

    auto graph = parse_matrix(matrix_str);
    int n = graph.size();
    if (n == 0 || graph[0].size() != n) {
        std::cerr << "Error: la matriz debe ser cuadrada y no vacía. Encontrado "
                  << n << "x" << (n>0 ? graph[0].size() : 0) << "\n";
        return 1;
    }

    std::vector<dist_t> dist(n, INF);
    std::vector<bool> visited(n, false);
    dist[src] = 0;

    for (int k = 0; k < n; ++k) {
        int u = -1;
        dist_t minDist = INF;
        #pragma omp parallel
        {
            int local_u = -1;
            dist_t local_min = INF;
            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                if (!visited[i] && dist[i] < local_min) {
                    local_min = dist[i];
                    local_u = i;
                }
            }
            #pragma omp critical
            {
                if (local_min < minDist) {
                    minDist = local_min;
                    u = local_u;
                }
            }
        }
        if (u == -1) break;
        visited[u] = true;

        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            if (!visited[v] && graph[u][v] > 0 && dist[u] < INF) {
                dist_t newDist = dist[u] + graph[u][v];
                #pragma omp critical
                {
                    if (newDist < dist[v]) dist[v] = newDist;
                }
            }
        }
    }

    std::ofstream ofs(out_path);
    if (!ofs) {
        std::cerr << "Error al abrir archivo: " << out_path << std::endl;
        return 1;
    }
    ofs << "Vértice\tDistancia desde el origen\n";
    for (int i = 0; i < n; ++i) {
        ofs << i << "\t" << (dist[i]>=INF ? -1 : dist[i]) << "\n";
    }
    return 0;
}
