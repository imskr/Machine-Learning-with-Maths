import numpy as np

class univariate_Linear_Regression:
    # initialize slope and intercept
    def __init__(self):
        self.m = 0.0  # slope
        self.b = 0.0  # intercept

# sum of square deviation for single variable
    def ss_x(self, x):
        return sum((x-np.mean(x))**2)

    # sum of square deviation for two variables x and y
    def ss_xy(self, x, y):
        x_mean = np.mean(x)			# mean of x
        y_mean = np.mean(y)			# mean of y
        return sum((x-x_mean)*(y-y_mean))

    # Train our regression model based on shape and size
    def train(self, x, y):

        # verify the features and labels are of same size
        assert(len(x) == len(y))

        # calculate the slope
        ss_x = self.ss_x(x)
        ss_xy = self.ss_xy(x, y)
        self.m = ss_xy/ss_x

        # calculate the intercept
        self.b = (np.mean(y)) - (self.m)*(np.mean(x))

    # return the predicted values based on feature and weights
    def predict(self, x):
        predictions = np.zeros(len(x))
        for i in range(len(x)):
            predictions[i] = self.m * x[i] + self.b         # Y = mx + b
        return predictions


# Dataset to train model
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.1, 1.9, 2.8, 4, 5.2, 5.8, 6.9, 8.1, 9, 9.9])

# Initialize our model
reg = univariate_Linear_Regression()

# Train our model with the data
reg.train(x, y)

# Make a prediction
print(reg.predict(x))