import Control.Applicative
import Language.Haskell.TH (safe, prim)
import System.Posix (QueueSelector(OutputQueue))
safeDivide :: Int -> Int -> Maybe Int
safeDivide _ 0 = Nothing
safeDivide x y = Just (x `div` y)
safeSquareRoot :: Int -> Maybe Int
safeSquareRoot x
 | x < 0 = Nothing
 | otherwise = Just (round . sqrt $ fromIntegral x)
complexOperation :: Int -> Int -> Maybe Int
complexOperation x y = safeSquareRoot =<< safeDivide x y
main :: IO ()
main = do
    print $ complexOperation 100 2 -- Output: Just 10
    print $ complexOperation 100 0 -- Output: Nothing