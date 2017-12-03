# Gradient Descent
- Know the purpose of gradient descent, and name some specific applications we already know.
  - In machine learning, we oftentimes want to minimize some loss (or cost) function. Loss functions include quadratic loss function and 0-1 loss function. Gradient descent is a method for finding the minimum of a loss function. If your cost is a function of K variables, then the gradient is the length-K vector that defines the direction in which the cost is increasing most rapidly. We follow the negative of the gradient to get to the point where the loss is a minimum.

  - Example application:
    ```
      # From calculation, it is expected that the local minimum occurs at x=9/4
      cur_x = 6 # The algorithm starts at x=6
      gamma = 0.01 # step size multiplier
      precision = 0.00001
      previous_step_size = cur_x

      df = lambda x: 4 * x**3 - 9 * x**2

      while previous_step_size > precision:
        prev_x = cur_x
        cur_x += -gamma * df(prev_x)
        previous_step_size = abs(cur_x - prev_x)

      print("The local minimum occurs at %f" % cur_x)
    ```

- Write pseudocode of the gradient descent and stochastic gradient descent algorithms.
  1. Choose a random starting point for your variables. For performance reasons, this starting point should really be random - use a pseudorandom number generator to choose it.
  2. Take the gradient of your cost function at your location.
  3. Move your location in the opposite direction from where your gradient points, by just a bit.
  4. Repeat steps 2 and 3 until you’re satisfied and repeating them more doesn’t help you too much.
  5. Using gradient descent with a few hundred iterations, we can easily find parameters for our model which give us a nice fit.

- Gradient Descent convergence criterion:
  1. Max number of iterations
  2. Change in cost function: `(costold−costnew)/costold<ϵ``
  3. Magnitude of gradient

- Warning:
  1. Requires differentiable, convex function
  2. Only finds global optimum on globally convex function
  3. Converges linearly for strongly convex functions.
  4. May not converge for weakly convex functions.
  5. Requires feature scaling
  6. Learning rate must be chosen (well)

- Compare and contrast batch and stochastic gradient descent - the algorithms, costs, and benefits.
  - from a dat ingestion perspective: in full batch gradient descent algorithms, we use whole data at once to compute the gradient, whereas in stochastic we take a sample while computing the gradient.

- Draw Gradient Descent cartoons

- To Do:
https://www.analyticsvidhya.com/blog/2017/03/introduction-to-gradient-descent-algorithm-along-its-variants/
