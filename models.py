import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
  def __init__(self, num_inputs, learning_rate=0.01):
    self.weights = np.random.rand(num_inputs+1)
    self.learning_rate = learning_rate

  def weighted_sum(self, inputs):
    Z=np.dot(inputs, self.weights[1:])+self.weights[0]
    return Z

  def predict(self, x):
    z=np.dot(x, self.weights[1:])+self.weights[0]
    return z
    
  def loss(self, prediction, target):
    return np.mean((prediction-target)**2)
  
  def fit(self, X, y, tolerance=10e-5,  n_epochs = 100):
    self.history = []
    for _ in range(n_epochs):
        for x, y_s in zip(X, y):
          y_pred = self.predict(x)
          err = (y_s - y_pred)

          mse = self.loss(y_pred, y_s)
          if mse < tolerance:
              return self.history
          
          change_w = -2 * err * x
          change_b = -2 * err

          w_new = self.weights[1:] - self.learning_rate * change_w
          b_new = self.weights[0] - self.learning_rate * change_b
          self.weights[1:] = w_new
          self.weights[0] = b_new
          self.history.append(mse)
    return self.history


if __name__=="__main__":

    # from sklearn.datasets import make_regression
    
    # X, y = make_regression(
    #     n_samples=1000,
    #     n_features=1,
    #     n_targets=1,
    #     random_state=42
    # )
    # nn = Perceptron(1)
    # nn.fit(X, y)
    # pred = nn.predict(X)
    # print(nn.loss(pred, y))

    k = 5
    b = 3
    X = np.linspace(-10, 10, 1000)
    y = k * X + b
    error = np.linspace(-10, 10, 1000)
    np.random.shuffle(error)
    y_synt = y + error
    plt.plot(X, y_synt, "o", c = "r")
    plt.plot(X, y)

    plt.show()
