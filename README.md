# Words Embeddings

## Dependencies
* Python3
* Pytorch
* Tensorboard
* matplotlib
* seaborn

## Dircetory

* data: Contain the dataset
* report: latex code for the report
* src: Python source code
* SGNS.ipynb : Driver code for Skip-gram with negarive Sampling
* Vanilla_Skip_gram.ipynb : Driver code for vanilla skip gram

The stop words list can be downloade from : https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.json
and should be placed in the data folders, in order to run all the code.

**PS**: All the code were tested on a MacBook Pro M3 using `torch.mps` backend, so there might be some incompatibility with `cuda` or PC `cpu` (I didn't have time to test it on  PC, but I tried to implent it as sdndard as possible)
