---
title: "Review and comparison of two manifold learning algorithms: t-SNE and UMAP"
tags:
  - Machine-learning
  - AI
  - Data-Science
  - Data
  - Data-Centric-AI
date: 2022-05-05
---


## Manifold learning algorithms (MLA)

For us humans, high-dimensional data are very difficult to visualize and reason with. That's why we use dimensionality reduction techniques to reduce data dimensions so they are easy to work with. Manifold learning algorithms (MLA) are dimensionality reduction techniques that are sensitive to non-linear structures in data. The non-linearity is what sets manifold learning apart from other popular linear dimensionality reduction techniques like Principal Component Analysis (PCA) or Independent Component Analysis (ICA). Non-linearity allows MLAs to retain complex and interesting properties of data that would otherwise be lost in linear reduction/projection. Because of this property, MLA is a very handy algorithm to analyze data - to reduce data dimensions to 2D or 3D, and visualize and explore them to find patterns in datasets.

[t-SNE] (t-Distributed Stochastic Neighbor Embedding) and Uniform Manifold Approximation and Projection [UMAP] are the two examples of MLA, that I will cover in this post to compare and contrast and provide a good intuition of how they work and how to chose one over the other.
Note that, MLAs are tweaks and generalizations of existing linear dimensionality reduction frameworks themselves. Similar to linear dimensionality reduction techniques, MLAs are predominantly unsupervised even though supervised variants exist. In the scope of this post is unsupervised techniques, however.


So what does non-linearity buys us? The following shows the difference in reduction using PCA vs t-SNE, as shown in [McInnes] excellent talk:
>![](/images/feature-analysis/PCA_vs_t-SNE_on_MNIST.jpg)
Evidently, while PCA retains some structure. However, it is very well pronounced in t-SNE, and clusters are clearly separated. 


### Neighbour graphs 
[t-SNE] and [UMAP] are neighbor graph technique that models data points as nodes, with weighted edges representing the distance between the nodes. Through various optimizations and iterations, this graph and layout are tuned to best represent the data as the distance is derived from the "closeness" of the features of the data itself. This graph is then projected on reduced dimensional space a.k.a. embedded space. This is a very different technique than matrix factorization as employed by PCA, ICA for example.


### t-Distributed Stochastic Neighbor Embedding (t-SNE)

[t-SNE] uses Gaussian joint probability measures to estimate the pairwise distances in the data points in the original dimension. Similarly, the student's t-distribution is used to estimate the pairwise distances in embedded space (lower dimension, target dimension). t-SNE then uses gradient descent technique to minimize the divergence between the two distributions in original and embedded space using the Kullback-Leibler (KL) divergence technique.

`Perplexity` in t-SNE is effectively the number of nearest neighbors considered in producing the conditional probability measure. A larger perplexity may obscure small structures in the dataset while small perplexity will result in very localized output ignoring global information. perplexity must be less than several data points then otherwise we are looking at getting a blobby mass, however, the more recommended range for perplexity lies between 5-50 with the larger the data, the larger the perplexity as a more general rule.

Because of the use of KL divergence, t-SNE preserves the local structure in the original space however global structure preservation is not guaranteed. Having said that, when initialization with PCA is applied, the global structure is somewhat preserved. 


Talking more about structures, the scale of distances between points in the embedded space is not uniform in t-SNE as t-SNE uses varying distance scales. That's why it is recommended to explore data under different configurations to tease out patterns in the dataset.
Learning rate and number of iterations are two additional parameters that help with refining the descent to reveal structures in the dataset in the embedded space. As highlighted in this great [distill][distill-tsne] article on t-SNE, more than one plot may be needed to really understand the structure of the dataset.
>![](/images/feature-analysis/tsne-topo.jpg)


t-SNE is known to be very slow with the order of complexity given by O(dN^2) where d is the number of output dimensions and  N is the number of samples. Barnes-Hut variation of t-SNE improves the performance [O(dN log N)] however Barnes-Hut can only work with dense datasets and provide at most 3d embedding space. The efficiency gain in Barnes-Hut is coming from gradient calculation in n log n time and uses approximation technique leading to about 3% error in nearest neighbor calculation.

Because of these performance implications, a common recommendation is to use PCA to reduce the dimension before applying t-SNE. This should be considered very carefully especially if the point of using t-SNE was to explore into non-linearity of the dataset. Pre-processing with linear techniques like PCA will essentially lose any non-linear structures if any.



### Uniform Manifold Approximation and Projection [UMAP]

[UMAP] is based on pure combinatorial mathematics that is well covered in the [paper][UMAP] and is also well explained by author McInnes in his [talk][McInnes]. Similar to [t-SNE], [UMAP] is also a topological neighbor graph modeling technique. There are several differences b/w [t-SNE] and [UMAP] with the main one being that umap retains not only local but global structure in the data.

There is a great post that goes into detail about [how UMAP works][umap-understanding]. High level, UMAP uses combinatorial topological modeling with the help of simplices to capture the data and applies Riemannian metrics to enforce the uniformity in the distribution. Fuzzy logic is also applied to the graph to adjust the probability distance if the radius grows. Once the graphs are built then optimization techniques are applied to make the embedded space graph very similar to the original space one. UMAP uses binary cross-entropy as a cost function and stochastic gradient descent to iterate on the graph for embedded space. Both t-SNE and UMAP use the same framework to achieve manifold projections however implementation details vary. [Oskolkov]'s post covers in great detail the nuances of both the techniques and is an excellent read.

UMAP is faster for several reasons, mainly, it uses random projection trees and nearest neighbour descent to find approximate neighbours quickly.
<!-- As shown in figure below, similar to t-SNE, UMAP also varies the distance density in the embedded space
>![](/images/feature-analysis/umap-math.jpg) -->

Heres an exaple of UMAP retaining both local and global structure in embedded space:
>![](/images/feature-analysis/umap-topo.jpg)


The uses cross-entropy is key 
>![](/images/feature-analysis/tsne-vs-umap.jpg)


>![](/images/feature-analysis/ce-umap.jpg)


### Comparison table 

| Characteristics | t-SNE | UMAP |
|-----------------|-------|------|
|     Computational complexity            |   O(dN^2) <br />(Barnes-Hut with O(dN log N) )   | O(d*n^1.14) <br />([emprical-estimates] O(dN log N))     |
|   Local structure preservation              |    Y   |  Y    |
|   Global structure preservation              |    N <br />(somewhat when `init=PCA`)   |  Y    |
|                 |       |      |
|   Cost Function               |   KL Divergence    |   Cross Entropy   |
|   Initialization              |   Random <br /> (PCA as alternate)    |  Graph Laplacian    |
|   Optimization algorithm              |   Gradient Descent (GD)     |   Stochastic Gradient Descent (SGD)    |
|   Distribution for modelling distance probabilities              |   Student's t-distribution    | family of curves (1+a*y^(2b))^-1     |
|   Nearest neighbors  hyperparameter        |    2^Shannon entropy   |   nearest neighbor k   |
|                 |       |      |
|                 |       |      |
|                 |       |      |






## how to do this for multi-label 







[t-SNE]: https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
[UMAP]: https://arxiv.org/abs/1802.03426
[tsne-illustrated]: https://www.oreilly.com/content/an-illustrated-introduction-to-the-t-sne-algorithm/
[distill-tsne]: https://distill.pub/2016/misread-tsne/
[tsne-paper]: https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
[tsne-author]: https://lvdmaaten.github.io/tsne/
[karapathy]: https://cs.stanford.edu/people/karpathy/cnnembed/
[datascienceplus]: https://datascienceplus.com/multi-dimensional-reduction-and-visualisation-with-t-sne/
[tricks-to-tsne]: https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0
[McInnes]: https://www.youtube.com/watch?v=nq6iPZVUxZU
[tsne-utube]: https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw
[emprical-estimates]: https://github.com/lmcinnes/umap/issues/8
[umap-understanding]: https://pair-code.github.io/understanding-umap/index.html
[Oskolkov]: https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668