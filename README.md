[![Build status](https://travis-ci.org/dataplayer12/Fly-LSH.svg?master)](https://travis-ci.org/dataplayer12)
# Paper
Code accompanying our [paper](https://arxiv.org/abs/1812.01844) **Improving Similarity Search with High-dimensional Locality sensitive hashing**

# Summary
We make three important contributions:
1. We present a new data independent approximate nearest neighbor (ANN) search algorithm inspired by the fruit fly olfactory circuit introduced by [Dasgupta et. al.](http://science.sciencemag.org/content/358/6364/793/tab-article-info). Named *DenseFly*, the proposed algorithm performs significantly better than several existing data independent algorithms on six benchmark datasets. (figures 2 and 3)
2. We prove several theoretical results about the original *FlyHash* as well as the proposed *DenseFly* algorithms. In particular, we show that *FlyHash* preserves rank similarity under any *Lp* norm and that *DenseFly* approximates a *SimHash* in very high dimensions at a much lower computational cost. (Lemmas 1 and 2)
3. We develop a multi-probe binning scheme for *FlyHash* and *DenseFly* algorithms, which are indispensable for practical applications of ANN algorithms. Remarkably, the proposed multi-probe binning scheme does not require additional computation over and above those used to create the high dimensional *Fly* or *DenseFly* hashes. Thus, the multi-probe versions of *FlyHash* and *DenseFly* result in a significant increase in mAP scores for a given query time. (figure 4)

# Code
The code for all the new algorithms described are present in one large file. Helper scripts to compare different algorithms will be added soon.
