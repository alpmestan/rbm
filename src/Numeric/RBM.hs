module Numeric.RBM where

import Control.Monad
import Control.Monad.IO.Class
import Math.Probable hiding (vector)
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Numeric.LinearAlgebra.HMatrix (norm_2)

data RBM = RBM
  { weights  :: !(Matrix Double)
  , weightsT :: !(Matrix Double)
  } deriving (Eq, Show)

pp :: RBM -> IO ()
pp (RBM w _) = disp 2 w

randMat :: Int -> Int -> RandT IO Double -> IO (Matrix Double)
randMat n p = mwc . fmap fromLists . replicateM n . listOf p

new :: Int -- ^ nb of visible units
    -> Int -- ^ nb of hidden units
    -> IO RBM
new nvisible nhidden = do
  w      <- randMat nvisible nhidden (normal 0 0.1)
  let w'  = asColumn (konst 0 nvisible) ||| w
      w'' = asRow (konst 0 (nhidden + 1)) === w'
  return $ RBM w'' (tr w'')


trainOnce :: Matrix Double -- ^ training data
          -> Double        -- ^ learning rate
          -> RBM
          -> IO RBM
trainOnce trainingData learningRate (RBM w wt) = do
  let dat            = asColumn (konst 1 n) ||| trainingData
      pos_hidden_act = dat <> w
      pos_hidden_p   = cmap logistic pos_hidden_act
  
  pos_hidden_st <- do
    randomM <- randMat n p (uniform 0 1)
    return $ step (pos_hidden_p - randomM)

  let pos_assocs = tr dat <> pos_hidden_p

      neg_visible_act = pos_hidden_st <> wt
      neg_visible_p   = cmap logistic neg_visible_act
      n'              = rows neg_visible_p
      neg_visible_p'  = asColumn (konst 1 n') ||| dropColumns 1 neg_visible_p
      neg_hidden_act  = neg_visible_p' <> w
      neg_hidden_p    = cmap logistic neg_hidden_act
      neg_assocs      = tr neg_visible_p' <> neg_hidden_p

      new_w           = (w +) . cmap (\x -> learningRate * x / fromIntegral n) $
                          pos_assocs - neg_assocs

      err             = norm_2 (dat - neg_visible_p')

  putStrLn $ "Error: " ++ show err
  return $ RBM new_w (tr new_w)

  where n = rows trainingData
        p = cols w

logistic :: Double -> Double
logistic x = 1 / (1 + exp (negate x))

trainN :: Matrix Double -- ^ training data
       -> Double        -- ^ learning rate
       -> Int           -- ^ epochs
       -> RBM
       -> IO RBM
trainN dat learningRate epochs = go epochs

  where go 0 rbm = return rbm
        go n rbm = do
          rbm' <- trainOnce dat learningRate rbm
          go (n - 1) rbm'

runVisible :: RBM
           -> Matrix Double      -- ^ sample rows of values for visible units
           -> IO (Matrix Double) -- ^ values of hidden units for the given visible units
runVisible (RBM w wt) inputData = do
  let dat        = asColumn (konst 1 n) ||| inputData
      hidden_act = dat <> w
      hidden_p   = cmap logistic hidden_act

  hidden_st <- do
    randomM <- randMat n p (uniform 0 1)
    return $ step (hidden_p - randomM)

  return $ dropColumns 1 hidden_st

  where n = rows inputData
        p = cols w

runHidden :: RBM
          -> Matrix Double      -- ^ sample rows of hidden units values
          -> IO (Matrix Double) -- ^ values of visible units for the given hidden ones
runHidden (RBM w wt) hiddenData = do
  let dat         = asColumn (konst 1 n) ||| hiddenData
      visible_act = dat <> wt
      visible_p   = cmap logistic visible_act

  visible_st <- do
    randomM <- randMat n p (uniform 0 1)
    return $ step (visible_p - randomM)

  return $ dropColumns 1 visible_st

  where n = rows hiddenData
        p = cols wt

daydream :: RBM
         -> Int                -- ^ number of samples we want to extract
         -> IO [Vector Double] -- ^ each row is a sample of the visible units
                               --   that the RBM has "daydreamed"
daydream (RBM w wt) nsamples = do
  sample0 <- fmap (fromList . (1:) . tail) . mwc $ listOf n (uniform 0 1)

  go sample0 nsamples

  where n = rows w
        p = cols w

        go :: Vector Double -> Int -> IO [Vector Double]
        go _v 0 = return []
        go v i = do
          let hidden_act = asRow v <> w
              hidden_p   = cmap logistic hidden_act

          hidden_st <- do
            randomV <- fmap vector . mwc $ listOf p (uniform 0 1)
            return $ step (head (toRows hidden_p) - randomV)
          
          let hidden_st' = asRow . vector . (1:) . tail . toList $ hidden_st
              visible_act = hidden_st' <> wt
              visible_p   = cmap logistic visible_act

          visible_st <- do
            randomV <- fmap vector . mwc $ listOf n (uniform 0 1)
            return $ step (head (toRows visible_p) - randomV)

          let res = vector . tail . toList $ visible_st
          fmap (res:) $ go visible_st (i - 1)

-- EXAMPLE

-- data given in:
-- http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/

-- Harry Potter, Avatar, LOTR, Gladiator, Titanic, Glitter
samples :: Matrix Double
samples = fromLists
  [ [ 1, 1, 1, 0, 0, 0 ]
  , [ 1, 0, 1, 0, 0, 0 ]
  , [ 1, 1, 1, 0, 0, 0 ]
  , [ 0, 0, 1, 1, 1, 0 ]
  , [ 0, 0, 1, 1, 0, 0 ]
  , [ 0, 0, 1, 1, 1, 0 ]
  ]

new_user :: Matrix Double
new_user = fromLists [ [ 0, 0, 0, 1, 1, 0 ] ]

hiddens :: Matrix Double
hiddens = fromLists
  [ [0, 0]
  , [0, 1]
  , [1, 0]
  , [1, 1]
  ]

test :: Double -> Int -> IO ()
test rate k = do
  rbm <- new 6 2

  pp rbm
  trained <- trainN samples rate k rbm
  pp trained
  putStrLn "-------"
  putStrLn "VISIBLE"
  res <- runVisible trained new_user
  disp 2 res
  putStrLn "-------"
  putStrLn "HIDDEN"
  res' <- runHidden trained hiddens
  disp 2 res'
  
  putStrLn "-------"
  putStrLn "DAYDREAM"
  daydream trained 10 >>= mapM_ print


main :: IO ()
main = test 0.1 10000
