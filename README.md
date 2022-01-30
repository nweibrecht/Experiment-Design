# Script to reproduce CosRec training Data

`compare_datasets.py` repeats the steps as described in the [CosRec Paper](https://arxiv.org/pdf/1908.09972.pdf), 
to see whether they fit the used training data in the [CosRec Repository](https://github.com/zzxslp/CosRec)

The script further computes and prints the statistics for the reproduced and the preprocessed data sets,
as they were presented in the paper.

## TO DO: Download data sets
The script expects the data to be in the following structure:

```
data
└───gowalla
│   │   original_source.txt (Original Gowalla Data Set)
│   │   test.txt (Test Gowalla Data in CosRec Repository)
│   │   train.txt (Train Gowalla Data in CosRec Repository)
│   
└───ml
    │   ratings.dat (Original MovieLens Data Set)
    │   test.txt (Test MovieLens Data in CosRec Repository)
    │   train.txt (Train MovieLens Data in CosRec Repository)
```

[Original Gowalla Data Set](https://snap.stanford.edu/data/loc-gowalla.html)

[Original MovieLens (ML-1M) Data Set](https://grouplens.org/datasets/movielens/1m/)