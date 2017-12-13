# K Nearest Neighbors (kNN):
- Be able to describe the KNN algorithm:
  - Answer:
    - KNN is a non-parametric approach for classification problems. It starts with storing all data points. Before prediction, we need to define a metric for distance. Distance metrics include Euclidean distance, Manhattan distance, cosine distance, etc. For each data point, we calculate the desired distance between this point and all the rest data from our set. We predict the label of this data point by taking the majority votes on the k-nearest points. We do this for all data points in our set. Typical k can be 5 or 10.
    - The weights of each vote can be scaled by the inverse of the pair distance, thus signing higher votes to the points that are nearer.
    - For regression problems, instead of votes, we apply mean of continuous target.

- Describe the curse of dimensionality:
  - Answer:
    - The curse of dimensionality describes the sparsity in available data, when dimensionality increases drastically.
- Recognize the conditions under which the curse may be problematic:
  - Answer:
    - In order to obtain a statistically sound and reliable result, the amount of data needed to support the result often grows exponentially with the dimensionality.
    - Organizing and searching data often relies on detecting areas where objects form groups with similar properties; in high dimensional data, however, all objects appear to be sparse and dissimilar in many ways, which prevents common data organization strategies from being efficient.
- Enumerate strengths and weaknesses of KNN:
  - Advantage:
    1. Robust to noisy training data (especially if we use inverse square of weighted distance as the “distance”).  
    2. Effective if the training data is large.
  - Disadvantage:
    1. Need to determine value of parameter K (number of nearest neighbors).
    2. Distance based learning is not clear which type of distance to use and which attribute to use to produce the best results.
    3. Computation cost is quite high because we need to compute distance of each query instance to all training samples.

- Room for improvement:
  - KD tree for faster generalized N-point problems.
  http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
  - `class sklearn.neighbors.KNeighborsClassifier`:
    - algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
    - leaf_size : int, optional (default = 30) Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
