Include your answers to this morning's exercises in `individual.py`.

This morning we will revisit some linear algebra using the numpy library in python. 

**For the following exercises, use numpy vector/matrix operations. Do not use a for loop unless given explicit instructions**

## Part 0: Important Numpy Notes:

In an numpy array, a **row vector** is defined as:

```python
a = np.array([[1, 2, 3]])
```
The shape of `a` is `(1, 3)`.

A **column vector** is defined as:
```python
b = np.array([[1], [2], [3]])
```
The shape of `b` is `(3, 1)`.

Check the `shape` of all the vectors throughout the exercise.
If the shape is missing a value, i.e. `(3,)` or  `(,3)`, use `np.newaxis` to
restore the correct dimensions.


## Part 1: Linear Algebra Practice:

### Part 1.1

The stochastic matrix is central to the Markov process. It is a square matrix specifying that probabilities of going from one state to the other such that every column of the matrix sums to 1.

The probability of entering a certain state depends only on the last state occupied and the stochastic matrix, not on any earlier states.

Suppose that the 2004 **state of land use** in a city of 60 mi^2 of built-up
area is:

```
In 2004:
   
C (Commercially Used): 25%
I (Industrially Used): 20%
R (Residentially Used): 55%
```

1. Find the **state of land use** in **2009** and **2014**,
   assuming that the transition probabilities for 5-year intervals are given
   by the matrix **A** and remain practically the same over the time considered.
   
   <div align="center">
      <img src="images/transition_matix_A.png">
   </div>
   
   
<br>

### Part 1.2

This following question uses the `iris` dataset. Load the data in with the following code.
   
```python
from sklearn import datasets
# The 1st column is sepal length and the 2nd column is sepal width
sepalLength_sepalWidth = datasets.load_iris().data[:, :2]
```
  
1. Make a scatter plot of sepal width vs sepal length
  
2. Compute the mean vector (column-wise) of the data matrix. The `shape`
   of the mean vector should be `(1, 2)`
     
   Plot the mean vector on the scatter plot in `1.` 

   <div align="center">
    <img src="images/mean.png">
   </div>

3. Write a function (`euclidean_dist`) to calculate the euclidean distance
   between two **column vectors (not row vector)**. Your function should check
   if the vectors are column vectors and the shape of the two vectors are the same .

4. Write a function (`cosine_sim`) to calculate the cosine similarity_between 
   two **column vectors (not row vector)**.
   
5. Write a function that would loop through all the data points in a given matrix and 
   calculate the given distance metric between each of the data point and the mean
   vector. **A for loop is allowed here**
      
   **Input of the function:**
     - Data matrix as an ndarray
     - Function to compute distance metric (Euclidean / Cosine Similarity)
      
   **Output of the function:**
     - An array shaped `(150, 1)`
      
   Use the function to compute Euclidean Distance and Cosine Similarity between each of
   the data points and the mean of the data points. You should be able to call the function
   in this manner:

   ```python
   euclidean_dists = compute_dist(sepalLength_sepalWidth, euclidean_dist)
   cosine_sims = compute_dist(sepalLength_sepalWidth, cosine_sim)
   ```
6. Plot histograms of the euclidean distances and cosine similarities.
   
   <div align="center">
    <img src="images/eucli_hist.png">
   </div>

   <div align="center">
    <img src="images/cos_hist.png">
   </div>


## Extra Credit: Implementing the PageRank Algorithm

The [Page Rank Algorithm](http://en.wikipedia.org/wiki/PageRank) is used by Google
Search (in their early days) to rank websites in their search engine in terms 
of the importance of webpages. 
[More about PageRank](http://books.google.com/books/p/princeton?id=5o_K4rri1CsC&printsec=frontcover&source=gbs_ViewAPI&hl=en#v=onepage&q&f=false)

We will implement PageRank on this simple network of websites.

   <div align="center">
    <img src="images/pageweb.png">
   </div>

**In the above image:**
   - Each node is a web page
   - Each directed edge corresponds to one page referencing the other
   - These web pages correspond to the states our Markov chain can be in
   - Assume that the model of our chain is that of a random surfer/walker.

In this model, we transition from one web page (state) to the next with
equal probability (to begin).  Or rather we randomly pick an outgoing edge
from our current state.  Before we can do any sort of calculation we need to
know how we will move on this Markov Chain.

1. Create an `numpy ndarray` representing the transition probabilities between
   nodes for **the above network (in the image)**. The position _i_, _j_ in the matrix corresponds to the
   probability of going from node _i_ to node _j_.

2. Now that we have a transition matrix, the next step is to iterate on this
   from one page to the next (like someone blindly navigating the internet) and
   see where we end up. The probability distribution for our random surfer can
   be described in this matrix notation as well (or vector rather).

   Initialize a vector for the probability of where our random surfer is.
   It will be a vector with length equal to the number of pages.
   Initialize it to be equally probable to start on any page
   (i.e. you start randomly in a state on the chain).

3. To take a step on the chain, simply matrix multiple our user vector by the
   transition matrix.
   After one iteration, what is the most likely location for your random surfer?

4. Plot how the probabilities change.
   Iterate the matrix through the first ten steps.
   At each step create a bar plot of the surfers probability vector.

5. This time to compute the stationary distribution, we can use numpy's
   matrix operations. Using the function for calculating [eigenvectors](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) compute the
   stationary distribution (page rank).  Is it the same as what you found
   from above?  What is it's eigenvalue?
   
   **Hint:** 
   - The stationary state is represented by the real form of the left (first) eigenvector
   - The left eigenvector obtained from `numpy.linalg.eig` has to be normalized
           
