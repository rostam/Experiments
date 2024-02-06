module Main (main) where

import Data.Graph.Inductive
import Data.List (nub, sort, (\\))
import qualified Data.Map as Map
import Lib

-- main :: IO ()
-- main = someFunc


-- Represents the graph coloring result: a mapping from node to color.
type Coloring = Map.Map Node Int

-- Greedy graph coloring algorithm.
greedyColoring :: (Graph gr) => gr a b -> Coloring
greedyColoring gr = foldl colorNode Map.empty (nodes gr)
  where
    colorNode colMap n = 
      let adjColors = map (\adjN -> Map.findWithDefault 0 adjN colMap) (suc gr n ++ pre gr n)
          newColor = head ([0..] \\ adjColors) -- Find the lowest unused color.
      in  Map.insert n newColor colMap

-- Example usage with a simple undirected graph.
exampleGraph :: Gr () ()
exampleGraph = mkGraph [(1, ()), (2, ()), (3, ()), (4, ())] [(1,2,()), (2,3,()), (3,4,()), (4,1,()), (1,3,())]

main :: IO ()
main = do
  let coloring = greedyColoring exampleGraph
  print coloring



