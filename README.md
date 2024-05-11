# Optimization-Algorithms-Implementation
# LinearRegression 
The LinearRegression class is designed for linear regression tasks, providing methods for fitting the model, making predictions, and evaluating the model's performance.

## Methods
- `fit`: Fits the linear regression model to the training data.
- `predict`: Predicts outcomes based on input data.
- `score`: Computes the model's score or performance metric.

## Attributes
- Number of iterations: Number of iterations used during model fitting.
- MSE: Mean Squared Error, a measure of model accuracy.
- Parameters (thetas): Coefficients or weights learned by the model.
- Cost_history: History of cost function values during training.
- Parameters_history: History of parameter values during training.

## Hyperparameters
- Epochs: Number of training epochs.
- Learning rate: Rate at which the model learns from data.
- Optimizer: Optimization algorithm used (options: Batch gradient descent, mini_batch, stochastic with momentum, Nag, Adagrad, RMS, Adam).
- Batch size: Number of samples used in each iteration (for batch or mini-batch optimizers).
- Gamma: Momentum parameter for momentum-based optimizers.
- Gradient tolerance: Tolerance level for gradient convergence.
- Cost tolerance: Tolerance level for cost function convergence.
- Beta: Hyperparameter for certain optimizers (e.g., Adam).
- Beta_2: Hyperparameter for certain optimizers (e.g., Adam).
- Epsilon: Hyperparameter for certain optimizers (e.g., Adam).

## Optimizers
- Batch gradient descent
- Mini-batch gradient descent
- Stochastic gradient descent with momentum
- Nesterov Accelerated Gradient (NAG)
- Adagrad
- RMSprop (RMS)
- Adam

---

# BFGS Optimizer Algorithm 
The BFGS optimizer algorithm class implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm for optimization tasks.

## Methods
- `fit`: Fits the optimization model to the training data.
- `predict`: Predicts outcomes based on input data.
- `score`: Computes the model's score or performance metric.

## Attributes
- Number of iterations: Number of iterations used during optimization.
- MSE: Mean Squared Error, a measure of optimization accuracy.
- Parameters (thetas): Optimized parameters or coefficients.

## Hyperparameters
- Iterations: Number of optimization iterations.
- Learning rate: Rate at which the optimization algorithm updates parameters.
- Gradient tolerance: Tolerance level for gradient convergence.
- Cost tolerance: Tolerance level for cost function convergence.

