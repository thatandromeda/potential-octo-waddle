# Machine Learning

## Lesson 8

### k-means algorithm
* the steps:
  * pick k centroids (randomly)
    * k < m (where m is the number of training examples)
    * randomly pick k of the training examples as starting centroids
    * usually this works but if you have unlucky original points you can end up at bad local optima
    * avoid this problem by running it several times (50-1000) with different initializations
    * if k is small this will help a lot
    * if k >> 10 you are unlikely to do much better than your first pick
  * for each data point, label it with the centroid it is closest to
  * for each set, choose a new centroid which is the average of all data points in that set
  * so at any given point you are keeping track of two variable sets:
    * c(i), the index of the cluster to which example x(i) is assigned
    * mu(k), the centroid of cluster k
      * therefore mu(c(i)) = the centroid of the cluster to which example x(i) has been assigned
* what happens when the data are not well-separated?
  * (this section is unsatisfying. but)
  * if you need there to be n populations, this gives you a way to choose them
  * you can then examine each one separately to learn what makes most sense for it
* optimization objective
  * minimize the sum of the squared distance between each example and its cluster centroid
  * the centroid-moving step achieves this

## Lesson 7

### Support vector machines!
* popular, powerful supervised learning algorithm
* Sometimes uses C on the first term rather than lambda on the second - it doesn't really matter since multiplying by a constant doesn't change your minimization problem (therefore setting C to 1/lambda yields the same result as if you regularized with lambda rather than C)

### Large margin classifiers
* Sometimes it's not good enough to be a little above a threshold - we want to be a lot to either side of it in order to classify
* We want SVMs to choose decision boundaries that really maximize the separation between our examples (i.e. that have the largest possible minimum distance from all examples)
  * this distance is called its "margin"
* If C is very large, the SVM is quite sensitive to outliers - will redraw its function substantially to account for them. But if C is reasonably small, outliers will have only a small effect on the decision boundary (they may even be on the wrong side of it if they are extreme outliers)

### Math of large margin classifiers

Math reminders:
* u(transpose)v = inner product of vectors u and v
* ||u|| = norm of u = length of vector
  * for 2d vectors, this is just the Pythagorean theorem
* u(transpose)v = p * ||u||, where p is the length of the projection of v onto u

### SVMs in practice

* Don't write your own solver
* But do choose your parameter C and your kernel
  * you may need to implement the kernel
* Kernel examples:
  * no kernel ("linear kernel")
    * predict y=1 if Theta-transpose x >= 0
    * good choice for a huge number of features and small training set because you will definitely overfit in that case if your function is complicated
  * Gaussian kernel
    * in this case you must also choose sigma
    * good when n small and m large (few features, many training examples)
    * do feature scaling before using this kernel if the features are on very different scales - otherwise the large features will dominate your calculation (visualize the Gaussian curve, how skewed it would have to be to accommodate those differences in an egalitarian way)
  * Not all similarity functions are valid kernels
    * Must satisfy "Mercer's Theorem", whatever that is
    * This allows numerical solvers to work efficiently
    * Also means they don't diverge
    * Therefore people generally use the linear or Gaussian kernel, although you don't have to
    * Other options:
      * polynomial kernel
        * (similarity  = (X-transpose l) squared, or cubed, or possibly plus a constant and then exponentiated)
        * Usually performs worse than Gaussian
        * and only for nonnegative things, so that if you cube it it doesn't get weird
      * String kernel (for text strings)
      * chi-square
      * histogram intersection...
      * These are extremely unlikely to be the solution to your problem

* What if you want to do multiclass classification?
  * Some systems have this functionality built in
  * Or: use one-vs-all method; train K SVMs to distinguish your K classes

* Deciding among logistic regression and SVMs
  * if your number of features is large relative to your number of training examples:
    * logistic regression, or SVM with linear kernel
  * if n is small (1-1000) and m is intermediate (10-10,000)
    * SVM with Gaussian kernel
  * n is small but m is large (50,000+)
    * create or add more features; then use logistic regression or an SVM without a kernel
    * computationally SVMs may struggle on large training sets
  * logistic regression and SVMs without kernels are similar; differences among them are more computational than classificatory performance

## Lesson 6

### Evaluating a learning algorithm

Imagine that you've trained an algorithm, but it's making large errors on your testing set. What can you try?
* get more training examples
  * not always helpful, though
  * and always time-consuming
* try a smaller feature set to avoid overfitting
* or: collect more data to use -more- features
  * again, it would be nice to know in advance if this is useful so as not to spend too much time on it
* add polynomial features
* change lambda

Some of these are time-consuming; gut feeling isn't the best way to approach the question of how to improve your algorithm.

Luckily, there are machine learning diagnostics we can run to get insight into what is & isn't working and what might help. They are time-consuming, but not as much so as spending months gathering data you don't need.

### Evaluating a hypothesis

How can you tell if a hypothesis is overfitting?
* past two or three dimensions, plotting the hypothesis function is no longer an option
* split the data into a training and a test set
  * typically we do 70% training/30% test
  * if the data have any ordering, make sure to split them randomly (i.e. randomize the order and then take the first 70%)

Training/testing procedure:
* learn theta from training data, minimizing J(theta)
* Compute test set error (using the same J function, whether linear or logistic)
* Alternatively, just use _misclassification error_ for logistic regression:
  * assign 0/1, success/failure, to each example
    * it's a success if it's <= 0.5 and was supposed to be 0, and so forth
  * and then average that

### Model selection and training/validation/test sets

Training sets are prone to overfitting, and this makes it impossible to predict how well your hypothesis will generalize to novel examples.

Imagine you're trying to choose among multiple models (e.g. polynomials of varying degree). Now, in effect, you have one additional parameter, representing which model you choose.

However, the size of the error on the test set is not a good estimate of the error elsewhere due to overfitting, which means choosing the model with the lowest error is not a reliable way to pick the most generalizable model.

So, to select a modeL:
* split your data into _three_ sets:
  * ~60% training
  * ~20% cross-validation
  * ~20% test
* train on the training set
* select your model _on the cross-validation set_
* and now your test set will let you gauge the generalization error

### Diagnosing bias vs variance

* Bias = _underfitting_ the data (tending too strongly toward an incorrect hypothesis)
* Variance = _overfitting_ the data

As we increase the degree of a polynomial, the training set error typically decreases. However, the cross-validation error will increase if the model is bad.

-> In a high _bias_ problem, the training set and CV set errors are both high
-> But in a high _variance_ problem, the training set error is low but the CV set error is high

It's not clear what "low" and "high" are exactly, but in an example of a misclassification problem (error is in [0, 1]), he gave 0.1 as low and 0.3 as high.

### Regularization and bias/variance

We've seen regularization as a tool for avoiding overfitting; let's dig in more.

How should you choose the regularization parameter lambda?
* have a range of options to try
  * he steps up in multiples of 2; 0.01, 0.02, 0.04, etc., up until about 10
  * pick the one that yields the lowest error on the CV set

As lambda goes up, the theta parameters diminish, so training set error increases as underfitting becomes a problem. CV error is high when lambda is too low or too high.

### Learning curves

Plotting them helps you diagnose whether your model has problems.

Train your model on just a handful of examples (like, 3) and plot J. As the training set gets larger, the hypothesis will become less of a fit - average training error increases.

Plot the training set average error against the number of training examples.

Similarly, check these models against your J(CV) - this should be quite high for small training sets but it should decrease as the training set increases. So....

* ideal case
  * the two curves tend both tend toward modest error, though not converging
* high bias
  * CV error will be high for a small number of examples, decrease slightly as m increases, but plateau pretty high
  * training error will be small at first but rapidly get high
  * they will converge at high error
  * in this situation, additional training data does not help
* high variance
  * training error starts low, increases slightly, but remains low
  * CV error starts high, decreases slightly
  * gap between the curves remains large
  * But more training data will help, as the CV error is slowly decreasing over time

### Deciding what to do next, part II

* When variance is high:
  * Get more training examples
  * Try a smaller set of features
  * Increase lambda
* When bias is high:
  * Try getting additional features
  * Add polynomial features
  * Decrease lambda

How does this relate to neural network architecture?
* Small neural networks (i.e. relatively few hidden layers/nodes)
  * this means few parameters
  * more prone to underfitting (= high bias)
  * computationally cheaper
* Large neural networks (i.e. more hidden layers and/or nodes)
  * more parameters
  * more prone to overfitting (= high variance)
  * can be computationally expensive
  * larger is often better, but benefit from regularization to address overfitting

-> large but regularized neural networks are often the best choice if computationally feasible

As ever, you can test it out by having train/test/CV sets and running them on several architectures

### Prioritizing what to work on

Example: spam classifier.
* how do you want to represent the features of the email?
  * perhaps: choose a list of 100 spammy & not-spammy words
    * although in practice, people pick the most frequent 10-50,000 words in a set and make them the features
  * define a feature vector x, which has a 0 or 1 for each word in the list

* how can you lower your error rate?
  * collect lots of data
    * e.g. honeypots for spam
    * not always helpful, though
  * develop more sophisticated features
    * e.g. use header information for spam
    * use stemmatization, collapse misspelled versions of a term into the canonical one, and other systems for combining tokens
  * brainstorm lots of possibilities and test them
    * don't fixate on your first idea

### Error analysis

Plan of attack:
* start with a simple algorithm you can implement quickly
* plot learning curves to see where you should be spending your time
* look at examples in cross-validation set that your algorithm performed poorly on - do you see anything interesting?
  * are there categories you tend to fail on
  * are there features that would help you predict those categories
  * especially for the categories you fail on most often
* have a metric - something that gives you a specific real number - you can use to evaluate your algorithm
  * for instance, maybe you're trying to determine whether stemmatization is useful
  * cross-validation error is often a useful metric
  * single real numbers are much faster than manually examining examples - this will let you make quick decisions

### Handling skewed classes

1% error is good if your data set is 50/50...but terrible if you're looking for something that happens rarely!
* single real number metrics are bad for skewed data - simply always predicting False for something that is False 99.9% of the time will give you very good performance on your single-value metric, yet be terrible

Enter....(drumroll)...precision/recall!

Categorize your results into true positives, false positives, true negatives, false negatives.

* precision = True positives/predicted positives (=true + false positives)
* recall = True positives/actual positives (=true positives + false negatives)

If you're dealing with skewed data, this will ensure you're not just optimizing for the common case; you can't get both precision and recall to be good by doing that.

By convention, y=1 for the rare case with skewed data (e.g. cancer is rare but the test turning up positive is y=1).

### Trading off precision and recall

You can set the logistic threshold somewhere other than 0.5 if avoiding false positives is more important than avoiding false negatives (or vice versa).

You can actually plot your precision vs recall and use that to determine good threshold values.

## Lesson 5

Cost function for neural network = sum of cost function for each layer

As before, to get layer n + 1, multiply layer n by your theta parameters and then take the sigmoid.

The error of each node is the actual value, minus what the value should be. This is pretty obvious for nodes in the last layer...how do we back-propagate it?

Error for layer n-1 = theta(n-1)' * error(n) .* (derivative of sigmoid)(previous layer values)
* the derivative turns out to be a.* (1-a), where a is the activation values for the layer (before the sigmoid has been applied)

There's no error for layer one, because that's our training set - it's definitionally accurate.

Via a "surprisingly complicated" derivation:
* partial derivative with respect to theta(i,j) of the cost function is alpha(j) * delta (i)

Backpropagation algorithm:
* set Delta(i,j for layer l) to 0 for each i, j, l, and all the first-layer alphas to your training data
* forward-propagate to compute all the alphas in all the layers
* compute the delta for the last layer, then the second-to-last, et cetera
* update all the Deltas: Delta(i,j) := Delta(i,j) + alpha(j for l)* delta(i for l+1)

## Lesson 4

Neural nets address a problem of: what if you have a nonlinear hypothesis, and a very large number of features?
* You can use the large number of features to model the nonlinearity
* But it becomes computationally intractable

Inspired by the idea of human brains as general-purpose learning machines

### Model representation

Layer 1: x inputs.
Layer 2: apply your theta parameters to x and then take the sigmoid function, to determine which neurons are actually activated. Also add one bias neuron (like our theta-zero), activated.
Layer n: ...et cetera...

You can have lots of different architectures by varying the number of hidden layers and the number of neurons in each level.

## Lesson 3

### Classification problems
* Classification problems
  * spam/not spam (email)
  * fraudulent/not fraudulent (credit card transactions)
  * benign/malignant
  * All problems of assigning 0 or 1 values; the 'negative class' and the 'positive class'
    * Usually mapping to absence and presence of a fact of interest
  * There are also problems that are nonbinary ('multiclass classification')

* Threshold classifier
  * everything below a certain point is in one class, everything above is in the other
  * this is a very different shape from linear regression, and linear regression may be a very bad fit for classification problems
    * Addition of a single data point can radically change y = mx + b curves and therefore change the classification of other data points
    * Threshold classifiers do not have this property

-> 'logistic regression'
* which is a classification algorithm, and not like linear regression at all
* we use it when 0 <= h(x) <= 1, i.e. on classification problems

Core concept: linear regression is a continuous-valued function, but classification is a discrete-valued problem. Therefore linear regression is a poor choice for it and we need a function with more nearly discrete outputs.

### Hypothesis representation

h(x) = g(Theta' * x)
* where g is the _sigmoid function_ or _logistic function_, g(z) = 1/(1 + e^-z)
* this function gives _probability estimates_
  * e.g. an output of 0.7 = a 70% chance that the correct category is 1 (the positive category)

### Decision boundaries

* above some h(x), we should just round up and say we're in category 1; below, in category 0
* this is the decision boundary
* this value of h(x) corresponds to some value of Theta-transpose * x
  * for instance, h(x) = 0.5 when Theta-transpose * x = 0
* note that it is a property of the _hypothesis_ and the _parameters_, not of the data set
  * (your decision boundary might be a bad choice if it doesn't match the data, but it's still the boundary)
* what if the boundary is nonlinear?
  * e.g. inside and outside of a circle
  * (x_1)^2 + (x_2)^2 = 1 will give you a circle
  * in general, higher-order polynomials give you all sorts of squiggly possibilities

### Cost function
How do you choose parameters theta when using the logistic function?
* with linear regression, we used the mean squared error function
* but unfortunately for logistic regression it's not convex
  * so gradient descent can get stuck in local minima
* for logistic regression, use:
  * cost = -log(h(x)) if y = 1
  *      = -log(1-h(x)) if y = 0
    * good because cost = 0 if your hypothesis is right, but tends toward infinity the wronger you get

A more compact statement of the above:
* cost = -y log (h(x)) - (1-y) log (1-h(x))

A vectorized interpretation of the above:
h = g(Xθ)

J(θ) = (1/m) * (−Y' * log(h) − (1−Y)' * log(1−h))

We still have J(theta) = (1/m) sum from one to m (cost function)
And we still want to use partial derivatives to minimize it...
* ...and we still minimize it with the same gradient descent algorithm as before!
  * but h(x) has a different value (logistic rather than linear), so the gradient descent values end up different

A vectorized implementation of gradient descent for logistic regression:

θ : =θ − (α/m) * X'* (g(Xθ) - Y)

### Advanced optimization

Sure would be nice if we could compute logistic regression quickly.
* there are algorithms that can even pick a different learning rate on each step to converge faster
* but they are super complicated to understand
* you shouldn't implement them yourselves - use libraries
  * conjugate descent
  * BFGS
  * L-BFGS
* there's a noticeable difference between good and bad implementations

In Octave:
* fminunc(@costFunction, initialTheta, options)
  * where your costFunction should - given theta - compute the values of J and its gradient
  * options = optimset('GradObj', 'on', 'MaxIter', 100) or something like that
  * see https://www.coursera.org/learn/machine-learning/supplement/cmjIc/advanced-optimization for syntax specifics

** gonna need to review fminunc

### Multiclass classification

y can take on more than 2 values.
* so any given thing still only belongs to one class - we just have more than one of them
* that said, under the hood we write this as multiple separate binary classification problems
* and train a logistic regression classifier on each class
* and then, when we want to identify an object, we run all our classifiers on it and pick the most confident
* "one-vs-all" or "one-vs-rest"

### The problem of overfitting

"Bias", in this case, is the idea that the hypothesis function has a strong idea about the shape of the data, which is not corrected by evidence about the actual shape of the data. (...which is not so different from what bias is in reality, huh.)
* this leads to -underfitting-; our curve just doesn't look much like the training set

Overfitting, by contrast, fits the training set perfectly but doesn't generalize well to new data.

You may have data for a very large number of features, not all of which are really useful.
* Problems:
  * especially tricky if your feature set is large but your training data set is small
  * also leads to a problem of difficulty in plotting/visualizing your feature universe and training data
* Possible solutions
  * Option 1: throw out less important features
    * may throw out that turn out to be useful, though
    * and you need a rationale/algorithm for selecting them
  * Option 2: regularization
    * Reduce magnitude of parameters
    * Works well when each feature is a little bit useful

### Overfitting and the cost function

Regularization
* use small parameters
* the contribution of any given term is small
* smoother function; can be quite close to a function with different features

J(theta) = (1/2m) * [sum(1 to m) (h(x) - y)^2 + lambda * sum(1 to n)(theta_j)^2]
* we omit theta_0, the parameter of our null feature, on purpose
* lambda = "regularization parameter"
* we're capturing a _tradeoff_ here:
  * on the one hand, we want to minimize the error
  * on the other hand, we want to keep the parameters small
* If lambda is too large, you will be underfitting again
  * because you will have a bias toward your hypothesis being a horizontal line - just y = theta_0

--> changing _parameters_ is a substitute for changing your _hypothesis_; you don't need to take a firm position on which features are needed or what order they should be if you allow parameters to be near zero

### Regularization and linear regression

How do we update our thetas with gradient descent?
* theta_0: same as ever (since it's not included in the lambda term)
* theta_1 through n: same as before plus a new term for the lambda part
  * theta_j := theta_j - (alpha/m) * sum[1 to m](h(x)-y)*x_j + (lambda/m)* theta_j
  * =(1 - alpha*lambda/m) * theta_j + the sum
    * Note that (1 - alpha*lambda/m) must be less than one (in practice it is just a tiny bit less)
    * This means that it shrinks the paramter a little bit
      * ...and then subtracting the squared error reduces it more

There are vectorized forms in the notes - https://www.coursera.org/learn/machine-learning/supplement/pKAsc/regularized-linear-regression . They cannot readily be written here.

## Lesson 2

### Multivariate linear regression: overview
Woo yeah, multiple variables!
* n = number of features
* x^(i) = the input features of the ith training example
* x^(i)_j = value of feature j in the ith training example

So if you have a table of features and values, x^(i) is the ith row of that table, and x^(i)_j is the value of the jth cell in that row.

This means our old y=mx+b hypothesis form no longer works; instead, h(x) = b + theta_1*x_1 + theta_2*x_2, et cetera..._
* So still solidly *linear*, but multivariate. Just like the title says.

If you set x_0 = 1, then you can make an n+1-dimensional vector of x values and an n+1-dimensional vector of theta values (for your space of n variables). Then (theta-vector)transpose * (x-vector) = your hypothesis. Woo yeah!

And you can still have a k x n matrix, X, in which each of your x cases is a row, so you can multiply the whole thing by the theta-vector at once.

### Multivariate linear regression: feature scaling

Gradient descent converges more quickly if you can scale your features to take on a similar range of values.
* with very differnet values you end up with a very steep elliptical valley and things slide back and forth
* normalize everything to approximately [-1, 1].
  * it doesn't have to be perfect - we just want to avoid having an overly steep/skewed ellipse for our cost function.

We tend to go for _mean normalization_ - make your range of features have a mean near zero.
* replace x_i with x_i - mu_i
* and divide by max(x) - min(x)

### Multivariate linear regression: learning rate

To make sure gradient descent is working, plot the minimum of the cost function J(theta) against the # of iterations. It should asymptotically approach 0, or at least something small.
* in fact it should *monotonically* decrease. !
  * it is _mathematically provable_ that this is true for sufficiently small alpha
  * (this is obvious if you visualize it)
* also past a certain point it isn't really changing so you can stop and it's okay.
  * You can choose an epsilon at which to stop, although it's not obvious what this epsilon should be.

Troubleshooting
* J(theta) is increasing, or increases past a certain number of iterations
  * -> use smaller alpha
* J(theta) keeps bouncing around
  * -> use smaller alpha
* It's super slow
  * -> use bigger alpha
  * It's actually possible that too-large alpha converges slowly, but more likely your alpha is too small
* Plotting J(theta) for a range of values will show you the answer regardless
  * Try alpha = 0.0001, 0.01, 0.1, 1 - that will show you where you need to be, approximately

### Multivariate linear regression: feature choice and polynomial regression

* different features may allow for a better-fit model
* some data may look polynomial rather than linear...hm
  * feature scaling is really important as your higher-order variables will have much different ranges than your low-exponent ones
  * lots of choices - look for combinations of exponents that plausibly match the shape
  * we will eventually see automated ways to guess at feature selection (!)

(Note: this changes the *hypothesis function* but not the *cost function*, so our gradient descent still works - its local and global minimum remain the same. The *hypothesis* function may have local, non-global minima, but that is fine.)

### The normal equation
* Lets us solve for theta analytically rather than doing gradient descent iteratively
* Per calculus: take the derivative and set it equal to zero to solve in one step
  * since theta is an n + 1 dimensional vector, not a real number, we need to solve this for the partial derivative with respect to every theta_j

If X is your matrix of features (one set of features per row), and y is your vector of outputs:
  * (X-transpose * X)^-1 * X-transpose * y = theta
  * or in Octave: pinv(X' * X) * X' * y
  * does not require feature scalign!

* Advantages:
  * no need to choose alpha
    * Gradient descent may require multiple runs to tune alpha
  * no need to iterate
* Disadvantages
  * slow when the number of features is very large
    * because the cost of inverting a matrix is roughly n-cubed
    * no problem for 100 features, starts getting pretty dicey at 10000, etc.

* but wait - what if X-transpose X is noninvertible? (aka "singular" or "degenerate")
  * This is rare
  * Octave...
    * will handle it transparently if you use pinv - it will manage to compute theta even if it's noninvertible
    * the inv() function is a literal inverse
    * pinv() is "pseudo-inverse"
  * most common causes:
    * redundant features
      * e.g. if you had both size in feet and size in meters
      * delete them!
      * this will guarantee noninvertibility
    * too many features (m <= n - i.e. more features than examples)
      * this is not guaranteed to be invertible
      * you just don't have enough data, anyway, to fit that many parameters
      * so delete some, or regularize them (we'll see how later)

### How to submit assignments

Apparently you just submit() in Octave?? When you're in the directory with the appropriate files
* you should have a submission password somewhere

### Octave tutorial

It's so much faster to use Octave than Matlab that you should prototype in Octave and only port to Matlab if you need a production-y system.
* python/numpy or R will also work, but Octave is easier

~= is not equals (not !=)

&&, || are AND, OR

a ; at the end of a line suppresses print (but does not seem to matter otherwise)

0 is false, 1 is true

disp() = print()

strftime type formatting is available

Matrix: A = [ 1 2 ; 3 4] etc. (semicolon as row separator)
Vector: v = [1; 2; 3]

v = 1:0.1:2 will make a vector that goes from 1 to 2 in steps of 0.1

ones(m,n) will generate an m x n matrix that is all 1s
2*ones(m,n) will give you an mxn matrix of all 2s
rand(m,n) gives you a matrix of all random numbers :)
* there are related commands if you would like random variables drawn from a different distribution (e.g. randn() for Gaussian)

eye(n) will give you the nxn identity matrix ('eye' is a pun on I)

help x = documentation

pwd and ls are available

load() gets a file; save will save it

who <- variables in current session
clear <- clears them

A(:,2) gives you column 2 to the end
You can assign to it: A(:,2) = [10; 11; 12]

Use vectors rather than for loops whenever possible - they're faster

## Lesson 1

"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - Tom Mitchell

### Supervised learning
That is: we know what the correct output should be for our inputs

Two types of supervised learning problems:
* _Regression_ = output is a continuous value
  * Ex: "what will this house sell for"
* _Classification_ = discrete value outputs (categories)
  * Ex: "is this tumor malignant or benign"

_Support vector machines_ use sneaky math to let you feed an infinite (!) number of features to your classifier

### Unsupervised learning
"Here is the data set; can you find some structure?"

Example: clustering algorithms
* These can be used to cluster *literal machines* - figure out which machines in your data centers tend to talk and put them closer together!

### Programming environment
We'll use Octave (like Matlab but open source) because it's a much simpler learning environment than C++ or Python or what-have-you, because lots of relevant functions are simple calls rather than giant messes
* e.g. svd() = singular value decomposition is one function in Octave
* but a lot of code in other languages

You can learn the concepts much faster here even if it'll be (#sorrynotsorry) hard to port them later.

### Cost function
This is the measure of dθiscrepancy between your hypothesized function output and your training data
* 'how much does it cost to be wrong' I guess
* Usually we use square error - there are other options but this is the usual one

J(parameter set) = your cost function
If you plot this function and minimize it, that gives you the best-fit parameter set

### Gradient descent
A general method for finding the minimum of some function
* can be used for cost functions of any number of parameters although we'll use two-input examples
* Start with your parameters at some value and we're going to move them around bit by bit until morale improves
* (0,0) is a common initialization
* From your initial point, you look in all directions and pick the one where taking a little step gets you downhill fastest; lather, rinse, repeat until convergence
* Extremely sensitive to initial conditions - you can end up at very different local minima depending on where you start
* alpha, the _learning rate_, controls how big a step you take downhill
    * θ_j := θ_j - α(ẟ/ẟθ_j)J(θ_0, θ_1)
    * Update both thetas simultaneously - perform calculations for both and -then- reassign variables for both
    * You've either taken the step or you haven't; you're stepping diagonally, not one axis at a time
* Automatically takes smaller steps as you get close to a local minimum, even if you don't vary alpha - the tangent slope gets closer to zero
* "Batch" gradient descent = in every step you're considering the entire training data set
* There is also a normal method from linear algebra, which we'll see later - it doesn't require iteration, but it also doesn't scale as well
* Because the standard cost function is convex (upward??) it has only one global minimum and no other local minima, so gradient descent is guaranteed to converge if you don't make alpha too big

### Linear algebra review
* _dimension_ = number of rows x number of columns
  * Not multiplied - literally something like 4x2
  * |R^4x2 = the set of all matrices over the reals with dimension 4x2
  * A_ij = the i,j entry = in the ith row and jth column of matrix A
* A vector is a special case - an nx1 matrix
  * They come in both 1-indexed and 0-indexed, because of course they do
  * 1-indexed is more common in math, but 0-indexed is sometimes more useful in ML
  * Assume 1-indexed for this course
* Arithmetic
  * Addition/subtraction
    * Requires matrices of same dimension
    * Add (subtract) the i,jth entries from each matrix together to get i,jth of new
  * Scalar multiplication/division
    * Simply apply to each element of the matrix
  * Matrix multiplication
    * For an m x n matrix, multiplied by an n x 1 vector, apply the vector column to each row of the matrix, and sum (so apply the item in the jth row of the matrix to the things n the jth column, and then add the row across)
    * An m x n matrix times an n x 1 matrix yields an m x 1 matrix
    * To multiply by a multi-column matrix, just keep doing that - applying the jth column of matrix 2 to the ith row of matrix 1 yields the i,jth element of the result
      * an m x n matrix times an n x k matrix yields an m by k matrix

You could write either of the following:
`prediction matrix = data matrix x parameters vector`

```
for i in range(len(data)):
  prediction(i) = ...
```

But the former turns out to be more computationally efficient.

You can use matrices to test a whole lot of hypotheses at once - have a matrix of training data, and a second one where each column is a paramter set, and go to town.
* may be very computationally efficient - highly optimized libraries using multicore processing

* Properties of matrix multiplication
  * not commutative
  * yes associative

* Identity matrix
  * the one with 1s on the diagonals and 0s everywhere else
  * denoted as I (or I_nxn)
  * AxI = IxA = A (although the dimension of I may differ if A is not square)

* Inverses
  * Such that A(A^-1) = (A^-1)A = I
  * Defined only on *square* matrices
  * Not all square matrices have inverses
    * matrices lacking inverses are 'singular' or 'degenerate'

* Transposes
  * make the nth row of A into the nth column of B
