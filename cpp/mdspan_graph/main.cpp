#include <vector>
#include <limits>
#include <print>
#include <mdspan>

template <typename T>
requires std::is_arithmetic_v<T>
class Graph {
public:
    static constexpr T INF = std::numeric_limits<T>::max();

private:
    std::vector<T> adjacencyMatrix;
    std::mdspan<T, std::dims<2>> matrixView;
    size_t numVertices;

public:
    explicit Graph(size_t vertices) 
        : numVertices(vertices)
        , adjacencyMatrix(vertices * vertices, INF)
        , matrixView(adjacencyMatrix.data(), vertices, vertices) 
    {
        for (size_t i = 0; i < numVertices; i++)
            matrixView[i, i] = 0;
    }

    // Copy constructor
    Graph(const Graph& other)
        : numVertices(other.numVertices),
          adjacencyMatrix(other.adjacencyMatrix),
          matrixView(adjacencyMatrix.data(), numVertices, numVertices)
    {}

    // Copy assignment operator
    Graph& operator=(const Graph& other) {
        if (this == &other) return *this;
        numVertices = other.numVertices;
        adjacencyMatrix = other.adjacencyMatrix;
        matrixView = std::mdspan<T, std::dims<2>>(adjacencyMatrix.data(), numVertices, numVertices);
        return *this;
    }

    // Move constructor
    Graph(Graph&& other) noexcept
        : numVertices(other.numVertices),
          adjacencyMatrix(std::move(other.adjacencyMatrix)),
          matrixView(adjacencyMatrix.data(), numVertices, numVertices)
    {
        other.numVertices = 0;
    }

    // Move assignment operator
    Graph& operator=(Graph&& other) noexcept {
        if (this == &other) return *this;
        numVertices = other.numVertices;
        adjacencyMatrix = std::move(other.adjacencyMatrix);
        matrixView = std::mdspan<T, std::dims<2>>(adjacencyMatrix.data(), numVertices, numVertices);
        other.numVertices = 0;
        return *this;
    }

    // Destructor
    ~Graph() = default;

    void addEdge(size_t u, size_t v, T weight) {
        if (u >= numVertices || v >= numVertices)  
            throw std::out_of_range("Vertex index out of bounds");  
        if (u == v)  
            throw std::invalid_argument("Self-loops are not allowed");

        matrixView[u, v] = weight;
        matrixView[v, u] = weight;
    }

    [[nodiscard]] bool isConnected(size_t u, size_t v) const{
        if (u >= numVertices || v >= numVertices)  
            throw std::out_of_range("Vertex index out of bounds"); 
        return matrixView[u, v] != INF;
    }

    [[nodiscard]] std::mdspan<const T, std::dims<2>> getAdjacencyMatrix() const {
        return matrixView;
    }

    [[nodiscard]] size_t getNumVertices() const noexcept {
        return numVertices;
    }
};

template <typename T>
void printGraph(const Graph<T>& g) {
    size_t n = g.getNumVertices();
    auto matrix = g.getAdjacencyMatrix();

    std::print("Adjacency Matrix:\n");
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            T weight = matrix[i, j];
            if (weight == Graph<T>::INF)
                std::print("   ∞ ");
            else
                std::print(" {:3} ", weight);
        }
        std::println();
    }
}

int main() {
    Graph<int> g(5);

    g.addEdge(0, 1, 4);
    g.addEdge(0, 2, 8);
    g.addEdge(1, 2, 2);
    g.addEdge(1, 3, 6);
    g.addEdge(2, 3, 3);
    g.addEdge(3, 4, 5);
    g.addEdge(4, 0, 7);

    printGraph(g);

    std::print("\nIs node 1 connected to node 3? {}\n", g.isConnected(1, 3) ? "Yes" : "No");
    std::print("Is node 0 connected to node 4? {}\n", g.isConnected(0, 4) ? "Yes" : "No");
}
