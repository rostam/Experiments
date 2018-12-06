use std::collections::HashMap;
type AdjList = HashMap<i32, Vec<i32>>;
extern crate petgraph;
use petgraph::Graph;
use petgraph::visit::Dfs;
use petgraph::dot::{Dot, Config};
fn main() {
    let width1 = 30;
    let height1 = 50;


    let mut v = Vec::new();

    v.push((1,2));
    v.push((2,3));
    v.push((1,3));
    v.push((3,4));

    let adj_l = adj_list(v);

   /* procedure DFS-iterative(G,v):
      let S be a stack
      S.push(v)
      while S is not empty
        v = S.pop()
        if v is not labeled as discovered:
          label v as discovered
          for all edges from v to w in G.adjacentEdges(v) do
            S.push(w)
*/
    println!(
        "The area of the rectangle is {} square pixels.",
        area(width1, height1)
    );

    println!("{:?}", adj_l[&1]);
    println!("{:?}", adj_l);

    for v in &adj_l[&1] {
      println!("{}",v);
    }

   let mut graph = Graph::<&str, &str, petgraph::Undirected>::new_undirected();
   let pg = graph.add_node("petgraph");
   let fb = graph.add_node("fixedbitset");
   let qc = graph.add_node("quickcheck");
   let rand = graph.add_node("rand");
   let libc = graph.add_node("libc");
   graph.extend_with_edges(&[
      (pg, fb), (pg, qc),
      (qc, rand), (rand, libc), (qc, libc),
   ]);

    let mut dfs = Dfs::new(&graph,qc);
    while let Some(nx) = dfs.next(&graph) {
      // we can access `graph` mutably here still
      //graph[nx] += 1;
      println!("{}", graph[nx]);
    }


    // Create a new undirected graph, g
    let mut g = Graph::<u32, u32, petgraph::Undirected>::new_undirected();
    let w = 0;
    let mut v = Vec::new();

    // Add 10 vertices to G
    for i in 1..11 {
        v.push(g.add_node(0));
    }

    // Connect with 15 edges
    for i in 0..4 {
        g.add_edge(v[i], v[i + 1], w);
        g.add_edge(v[i], v[i + 5], w);
    }
    g.add_edge(v[0], v[4], w);
    g.add_edge(v[4], v[9], w);

    g.add_edge(v[5], v[7], w);
    g.add_edge(v[5], v[8], w);
    g.add_edge(v[6], v[8], w);
    g.add_edge(v[6], v[9], w);
    g.add_edge(v[7], v[9], w);

    // Print in graphviz dot format
    println!("{:?}", Dot::with_config(&g, &[Config::EdgeNoLabel]));

    for vv in g.node_indices() {
      for nv in g.neighbors(vv) {
        match g.node_weight(nv) {
          Some(col_nv) => {
            match g.node_weight(vv) {
              Some(col_vv) => println!("{} {}", col_nv, col_vv),
              None => println!("None")
            }
          }, 
          None => println!("None")
        }
      }
      println!("TEST");
    }
}

fn adj_list(elist: Vec<(i32, i32)>) -> AdjList {
    let mut adjacency_list = AdjList::new();
    for &(source, target) in elist.iter() {
        adjacency_list.entry(source).or_insert(Vec::new()).push(target);
    }
    adjacency_list
}


fn area(width: u32, height: u32) -> u32 {
    width * height
}

