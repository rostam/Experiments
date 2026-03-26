# Experiments Repository

A collection of programming experiments across multiple languages, exploring language features, algorithms, and libraries.

## C++

- **cpp-new-features** — Explores C++17/20 features: fold expressions, structured bindings, `std::variant`, `std::filesystem`, regex, and compile-time computation.
- **template_metaprogramming** — Compile-time Fibonacci with `if constexpr`, binary literal conversion via templates, and type introspection.
- **functional** — Functional programming patterns: higher-order functions, lambda expressions, currying, and algorithm composition.
- **nonhomo** — Graph homomorphism search using bitsets and constraint propagation with distance-based heuristics.
- **nn** — Simple neural network with sigmoid activation, weight initialization, and configurable layer sizes.
- **filesystem** — Demonstrates the C++17 `std::filesystem` API: path manipulation, directory creation, and path queries.
- **mdspan_graph** — Graph represented as an adjacency matrix using C++23 `mdspan`, with Floyd-Warshall shortest paths.
- **cgal_test** — Computational geometry with CGAL: 2D points, segments, and orientation predicates.
- **ball_gravity** — SFML physics simulation of a bouncing ball under gravity.
- **goes_to** — Demonstrates the `-->` ("goes to") trick: post-decrement combined with `>` as a readable loop countdown.
- **andorid_sdl** — SDL2 window and renderer setup as a cross-platform graphics bootstrap.

## Python

- **NonHomomorphism** — Finds graph homomorphisms using set partitions; checks independence and edge-respecting conditions. Includes triangle-free graph detection and graph6 format processing.
- **ComputerVision** — Image processing demos: RGB color selection, region extraction, and Canny edge detection with OpenCV.
- **GNN** — Graph Neural Network on the Karate Club graph using PyTorch and NetworkX.
- **TelegramSearch** — Telegram API client for listing groups/channels, retrieving messages, and processing Persian text content.
- **TF** — TensorFlow 1.x experiments: gradient descent optimization, eager execution, and higher-order derivatives.
- **partial_coloring** — Sparse linear algebra experiments: GMRES with ILU preconditioner, and power function convergence analysis.
- **pyMetis** — Graph partitioning with PyMetis: reads sparse matrices in Matrix Market format and partitions into clusters.
- **ollama** — Queries a local LLM through the `ollama` Python library.
- **python-code-samples** — Machine learning and data science collection:
  - Supervised learning for income prediction (`finding_donors.py`)
  - Customer segmentation with k-means (`CustomerSegments.py`)
  - PCA for facial recognition with eigenfaces (`PCA.py`)
  - Independent Component Analysis for audio source separation (`IndependentComponentAnalysis.py`)
  - Movie rating clustering (`kMeansClusteringMovieRating.py`)
  - Keras classification examples (XOR, student admissions, softmax/sigmoid networks)
  - MPI parallel computing (`mpi.py`)
  - Data preprocessing and cleaning utilities

## Rust

- **rust_max_flow** — Ford-Fulkerson max flow (Edmonds-Karp variant) with BFS augmenting paths and unit tests.
- **rust_gtea** — Graph construction with the `petgraph` crate: adjacency list building and DOT-format visualization.

## Haskell

- **graph-coloring-fgl** — Greedy graph coloring using the Functional Graph Library (FGL), mapping nodes to integer colors via `Data.Map`.
- **First** — Language basics: hello world, safe division, and square root with `Maybe` error handling.

## Julia

- **first.jl** — Math functions: sphere volume, quadratic equation solver, Unicode support.
- **strings_basics.jl** — String and character operations with Unicode and ASCII conversions.
- **dataframes.jl** — DataFrame creation, indexing, and column access using the `DataFrames` package.

## Go

- **hello.go** — Go fundamentals: structs, maps, closures, goroutines, `defer`, and periodic scheduling.

## Clojure

- **hello_world.clj** — Hello world demonstrating basic Clojure syntax.

## License

MIT
