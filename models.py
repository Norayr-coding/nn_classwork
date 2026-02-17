import numpy as np

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
        y_pred = self.predict(X)
        err = (y - y_pred).reshape(-1, 1)

        mse = self.loss(y_pred, y)
        if mse < tolerance:
            return self.history
        
        change_w = -2 * np.mean(X * err, axis = 0)
        change_b = -2 * np.mean(err)

        w_new = self.weights[1:] - self.learning_rate * change_w
        b_new = self.weights[0] - self.learning_rate * change_b
        self.weights[1:] = w_new
        self.weights[0] = b_new
        self.history.append(np.mean(mse))
    return self.history


if __name__=="__main__":

    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_targets=1,
        random_state=42
    )
    nn = Perceptron(5)
    nn.fit(X, y)
    pred = nn.predict(X)
    print(nn.loss(pred, y))