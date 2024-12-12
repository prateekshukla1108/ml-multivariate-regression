import numpy as np
import matplotlib.pyplot as plt

class MultivariateLR:
    """
    Multivariate Linear Regression implementation with Gradient Descent
    
    This class basically is a simple implementation of multivariate linear regression 
    This is inspired by Andrew Ng's Machine Learning course.
    """
    
    def __init__(self):
        """
        Initializing regression parameters.
        
        Attributes:
        - theta: Weight parameters for the regression model
        - X: feature matrix
        - y: target values
        """
        self.theta = None
        self.X = None
        self.y = None
    
    def feature_normalize(self, X):
        """
        Normalize features using mean normalization.
        
        Args:
        X(numpy.ndarray): Original feature matrix
        
        Returns:
        the normalized feature matrix
        mean of each feature
        standard deviation of each feature
        """
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        
        # Avoid division by zero
        sigma[sigma == 0] = 1
        
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma
    
    def compute_cost(self, X, y, theta):
        """
        calculating the cost function for linear regression.
        
        Args:
        X(numpy.ndarray): feature matrix
        y(numpy.ndarray): target values
        theta (numpy.ndarray): current model parameters
        
        Returns:
        computed cost (mean squared error)
        """
        m = len(y)
        predictions = np.dot(X, theta)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost
    
    def gradient_descent(self, X, y, theta, alpha, num_iters):
        """
        Perform gradient descent to learn theta.
        
        Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        theta (numpy.ndarray): Initial theta values
        alpha (float): Learning rate
        num_iters (int): Number of iterations
        
        Returns:
        numpy.ndarray: Optimized theta values
        list: Cost history for each iteration
        """
        m = len(y)
        theta = theta.copy()
        cost_history = []
        
        for _ in range(num_iters):
            predictions = np.dot(X, theta)
            error = predictions - y
            
            # Update theta simultaneously
            theta = theta - (alpha / m) * np.dot(X.T, error)
            
            # Record the cost
            cost_history.append(self.compute_cost(X, y, theta))
        
        return theta, cost_history
    
    def fit(self, X, y, alpha=0.01, num_iters=1500):
        """
        Fit the linear regression model.
        
        Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        alpha (float, optional): Learning rate. Defaults to 0.01.
        num_iters (int, optional): Number of iterations. Defaults to 1500.
        """
        # Add intercept term to X
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Normalize features
        X[:, 1:], self.mu, self.sigma = self.feature_normalize(X[:, 1:])
        
        # Initialize theta
        initial_theta = np.zeros(X.shape[1])
        
        # Perform gradient descent
        self.theta, self.cost_history = self.gradient_descent(X, y, initial_theta, alpha, num_iters)
        
        self.X = X
        self.y = y
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
        X (numpy.ndarray): Input feature matrix
        
        Returns:
        numpy.ndarray: Predicted values
        """
        # Normalize input features using stored mu and sigma
        X_norm = (X - self.mu) / self.sigma
        
        # Add intercept term
        X_norm = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
        
        return np.dot(X_norm, self.theta)
    
    def plot_convergence(self):
        """
        Plot the cost convergence over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history, 'b')
        plt.xlabel('Iterations')
        plt.ylabel('Cost J(θ)')
        plt.title('Convergence of Cost Function')
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Generate some sample data
    np.random.seed(42)
    
    # Create synthetic data
    m = 100  # number of training examples
    X = np.random.randn(m, 2)  # two features
    true_theta = np.array([1, 2, 3])  # true parameters
    
    # Create target values with some noise
    y = true_theta[0] + true_theta[1] * X[:, 0] + true_theta[2] * X[:, 1] + np.random.randn(m) * 0.1
    
    # Initialize and fit the model
    model = MultivariateLR()
    model.fit(X, y)
    
    # Print learned parameters
    print("Learned theta:", model.theta)
    
    # Plot cost convergence
    model.plot_convergence()
    
    # Make some predictions
    predictions = model.predict(X[:5])
    print("\nSample Predictions:")
    print(predictions)
    print("\nActual Values:")
    print(y[:5])

if __name__ == "__main__":
    main()
