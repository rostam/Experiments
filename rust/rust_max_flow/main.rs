use std::cmp;
use std::collections::VecDeque;

pub struct MaxFlow {
    graph: Vec<Vec<i32>>,
    size: usize,
}

impl MaxFlow {
    pub fn new(size: usize) -> Self {
        let graph = vec![vec![0; size]; size];
        MaxFlow { graph, size }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, cap: i32) {
        self.graph[from][to] += cap;
    }

    fn bfs(&self, start: usize, end: usize, parent: &mut Vec<Option<usize>>) -> bool {
        let mut visited = vec![false; self.size];
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(u) = queue.pop_front() {
            for (ind, &val) in self.graph[u].iter().enumerate() {
                if !visited[ind] && val > 0 {
                    queue.push_back(ind);
                    visited[ind] = true;
                    parent[ind] = Some(u);
                    if ind == end {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn max_flow(&mut self, start: usize, end: usize) -> i32 {
        let mut parent = vec![None; self.size];
        let mut max_flow = 0;
        while self.bfs(start, end, &mut parent) {
            let mut path_flow = i32::MAX;
            let mut s = end;
            while s != start {
                path_flow = cmp::min(path_flow, self.graph[parent[s].unwrap()][s]);
                s = parent[s].unwrap();
            }
            max_flow += path_flow;
            let mut v = end;
            while v != start {
                self.graph[parent[v].unwrap()][v] -= path_flow;
                self.graph[v][parent[v].unwrap()] += path_flow;
                v = parent[v].unwrap();
            }
        }
        max_flow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_flow() {
        let mut max_flow = MaxFlow::new(6);
        max_flow.add_edge(0, 1, 16);
        max_flow.add_edge(0, 2, 13);
        max_flow.add_edge(1, 2, 10);
        max_flow.add_edge(1, 3, 12);
        max_flow.add_edge(2, 1, 4);
        max_flow.add_edge(2, 4, 14);
        max_flow.add_edge(3, 2, 9);
        max_flow.add_edge(3, 5, 20);
        max_flow.add_edge(4, 3, 7);
        max_flow.add_edge(4, 5, 4);
        assert_eq!(max_flow.max_flow(0, 5), 23);
    }
}
