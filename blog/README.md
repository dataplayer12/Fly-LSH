[![Build status](https://travis-ci.org/dataplayer12/Fly-LSH.svg?master)](https://travis-ci.org/dataplayer12)

# Fly-LSH
An implementation of efficient LSH inspired by fruit fly brain
Reference: A neural algorithm for a fundamental computing problem [Science](http://science.sciencemag.org/content/358/6364/793/tab-article-info)

# Understanding
Please read my blog post explaining the difference between usual LSH and this algorithm [here](https://medium.com/@jaiyamsharma/efficient-nearest-neighbors-inspired-by-the-fruit-fly-brain-6ef8fed416ee)

# Usage

### Exploring differences between LSH and Fly-LSH
Follow notebook.ipynb

### Using Fly-LSH
`from flylsh import flylsh`

`flymodel=flylsh(inputs,hash_length,sampling_ratio,embedding_size) #inputs is a numpy array. Need not be zero centered`

`nearest_neighbors=flymodel.query(query_idx,num_NNs)`

`model_mAP=flymodel.findmAP(num_NNs,n_points)`
