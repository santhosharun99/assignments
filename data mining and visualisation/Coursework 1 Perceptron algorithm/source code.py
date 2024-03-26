import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def activation(z):
    """
    Calculates the sign of z and returns the result.

    Args:
    z: A numeric value representing the weighted sum of inputs and biases

    Returns:
    A float or integer representing the sign of z.
    """
    return np.sign(z)

class Perceptron:
    """A binary classification model based on the Perceptron algorithm.

    Parameters:
    -----------
    learning_rate: float
        The step size used to update the weights and bias during training.
    iterations: int
        The number of times to iterate over the entire training dataset.
    regularisation_coefficient: float
        The regularization parameter that controls the degree of weight decay.

    Attributes:
    -----------
    weights: array-like, shape (n_features,)
        The learned weights for the input features.
    bias: float
        The learned bias term.
    activation: function
        The activation function used to classify the input data.

    Methods:
    --------
    fit(X, y)
        Train the model on the input data and their corresponding target labels.
    predict(X)
        Predict the binary class labels for the input data.
    """

    def __init__(self, learning_rate, iterations, regularisation_coefficient):
        self.weights = None
        self.bias = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularisation_coefficient = regularisation_coefficient

    def fit(self, X, y):
        """Train the Perceptron model on the input data and their corresponding target labels.

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            The input data.
        y: array-like, shape (n_samples,)
            The corresponding target labels.

        Returns:
        --------
        self: object
            The fitted Perceptron model.
        """
        n_features = len(X[0])

        # Initializing weights and bias
        self.weights = np.zeros((n_features))
        self.bias = 0
        
        # Iterating until the number of iterations
        for epoch in range(self.iterations):
            
            # Traversing through the entire training set
            for i in range(len(X)):
                z = np.dot(X[i], self.weights) + self.bias # Finding the dot product and adding the bias
                y_pred = self.activation(z) # Passing through an activation function
                if y[i] * y_pred <= 0:
                    # Updating weights and bias
                    self.weights = (1 - 2 * self.regularisation_coefficient) * self.weights +  y[i] * X[i] * self.learning_rate
                    self.bias = self.bias + y[i] * self.learning_rate
                else:
                    # Updating weights with L2 regularization
                    self.weights = (1 - 2 * self.regularisation_coefficient) * self.weights
                    self.bias = self.bias
                    
        return self.weights, self.bias

    def predict(self, X):
        """Predict the binary class labels for the input data.

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_pred: array-like, shape (n_samples,)
            The predicted binary class labels.
        """
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

class OvRPerceptron:

    def __init__(self, learning_rate, iterations, regularisation_coefficient):
        """
        Initialize an OvRPerceptron object.

        Parameters:
        learning_rate (float): The learning rate for the Perceptron algorithm.
        iterations (int): The number of iterations to run the Perceptron algorithm.
        regularisation_coefficient (float): The regularization parameter for L2 regularization.

        Returns:
        None
        """
        self.learning_rate = learning_rate
        self.iterations = iterations  
        self.regularisation_coefficient = regularisation_coefficient 
        self.classifiers = []
        self.y_pred = []

    
    def fit(self, X, y):
        """
        Train an OvRPerceptron model on the input data.

        Parameters:
        X (array-like): The input data to train the model on.
        y (array-like): The target variable for the input data.

        Returns:
        None
        """
        # Convert labels to binary
        for class_label in np.unique(y):
            y_bin = np.where(y == class_label, 1, -1)
            perceptron = Perceptron(learning_rate=self.learning_rate, iterations=self.iterations, regularisation_coefficient=self.regularisation_coefficient)
            perceptron.fit(X, y_bin)
            self.classifiers.append(perceptron)
        
    def predict(self, X):
        """
        Use the trained OvRPerceptron model to make predictions on new input data.

        Parameters:
        X (array-like): The input data to make predictions on.

        Returns:
        array-like: The predicted target variable for the input data.
        """
        y_pred=[]
        for i in range(len(X)):
            classlist=[]
            for perceptron in self.classifiers:
                scores = np.dot(X[i], perceptron.weights) + perceptron.bias
                classlist.append(scores)
            pred_class = [np.argmax(classlist)]
            y_pred.append(pred_class)
        return np.array(y_pred)
    
"""
     loading the train and test data from the respective files using the pandas read_csv() function. 
     The delimiter parameter is set to "," and header parameter is set to None as there are no column names in the dataset.
"""
# Load train and test data
train_path = "train.data"
test_path = "test.data"

train_data = pd.read_csv(train_path, delimiter=",", header=None)
test_data = pd.read_csv(test_path, delimiter=",", header=None)

"""
Next, the code filters out the first class by creating a new dataframe where the fourth column of 
train_data is not equal to "class-3". This filtered data is then saved to a new file "filtered_train_data_12" using to_csv() function.
"""
print("\033[1m\033[4mBINARY CLASSIFICATION\033[0m\033[0m")
print('\n')
# Filter out class 3
filtered_data_12 = train_data[train_data[4] != 'class-3']
filtered_data_12.to_csv("filtered_train_data_12", index = False, header= False)
test_data_filtered_12= test_data[test_data[4] != 'class-3']
test_data_filtered_12.to_csv("filtered_test_data_12", index = False, header= False)

"""
Then, the filtered data is converted to NumPy arrays and the target variable is converted to numerical values using a 
dictionary that maps class labels to numerical values.
"""
traindata = np.array(filtered_data_12)
testdata = np.array(test_data_filtered_12)

target_var_dict_12 = {'class-1': 1, 'class-2': -1}

# Split into features and target variables
X_train, y_train_string = traindata[:, :-1], traindata[:, -1]
y_train_12 = np.array([target_var_dict_12[yi] for yi in y_train_string])

X_test, y_test_string = testdata[:, :-1], testdata[:, -1]
y_test_12 = np.array([target_var_dict_12[yi] for yi in y_test_string])

"""
Next, a perceptron model is instantiated with desired hyperparameters. The learning_rate is set to 1, iterations is set
 to 20, and regularisation_coefficient is set to 0. 
"""

# Instantiate Perceptron with desired hyperparameters
perceptron = Perceptron(learning_rate=1, iterations=20,regularisation_coefficient=0)

"""
The perceptron is then fitted to the training data using the fit() method.
X_train as x and y_train_12 as y in fit method.
"""
# Fit Perceptron to training data
perceptron.fit(X_train, y_train_12)

"""
The target variable is then predicted for the test data and train data using the predict() method.

"""

# Predict target variable for test data
y_pred_12 = perceptron.predict(X_test)
x_pred_12 = perceptron.predict(X_train)

"""
The performance of the model is then evaluated using the mean() and round() functions to 
calculate the accuracy. The accuracy is printed for each pair of classes.
"""

# Evaluate performance
accuracy_12test = np.mean(y_pred_12 == y_test_12)
accuracy_12train = np.mean(x_pred_12 == y_train_12)
print("\033[4mclass-1 vs class-2\033[0m")
print(f"test Accuracy : {round(accuracy_12test*100,2)}")
print(f"train Accuracy: {round(accuracy_12train*100,2)}")
print('\n')

"""
Next, the code filters out the first class by creating a new dataframe where the fourth column of 
train_data is not equal to "class-1". This filtered data is then saved to a new file "filtered_train_data_23" using to_csv() function.
"""
# Filter out class 1

filtered_data_23 = train_data[train_data[4] != 'class-1']
filtered_data_23.to_csv("filtered_train_data_23", index = False, header= False)
test_data_filtered_23= test_data[test_data[4] != 'class-1']
test_data_filtered_23.to_csv("filtered_test_data_23", index = False, header= False)

"""
Then, the filtered data is converted to NumPy arrays and the target variable is converted to numerical values using a 
dictionary that maps class labels to numerical values.
"""

traindata = np.array(filtered_data_23)
testdata = np.array(test_data_filtered_23)

target_var_dict_23 = {'class-2': 1, 'class-3': -1}
# Split into features and target variables
X_train, y_train_string = traindata[:, :-1], traindata[:, -1]
y_train_23 = np.array([target_var_dict_23[yi] for yi in y_train_string])

X_test, y_test_string = testdata[:, :-1], testdata[:, -1]
y_test_23 = np.array([target_var_dict_23[yi] for yi in y_test_string])

"""
Next, a perceptron model is instantiated with desired hyperparameters. The learning_rate is set to 1, iterations is set
 to 20, and regularisation_coefficient is set to 0. 
"""

# Instantiate Perceptron with desired hyperparameters
perceptron = Perceptron(learning_rate=1, iterations=20,regularisation_coefficient=0)

"""
The perceptron is then fitted to the training data using the fit() method.
X_train as x and y_train_23 as y in fit method.
"""

# Fit Perceptron to training data
perceptron.fit(X_train, y_train_23)

"""
The target variable is then predicted for the test data and train data using the predict() method.
"""

# Predict target variable for test data
y_pred_23 = perceptron.predict(X_test)
x_pred_23 = perceptron.predict(X_train)

"""
The performance of the model is then evaluated using the mean() and round() functions to 
calculate the accuracy. The accuracy is printed for each pair of classes.
"""

# Evaluate performance
accuracy_23test = np.mean(y_pred_23 == y_test_23)
accuracy_23train = np.mean(x_pred_23 == y_train_23)
print("\033[4mclass-2 vs class-3\033[0m")
print(f"test Accuracy : {round(accuracy_23test*100,2)}")
print(f"train Accuracy: {round(accuracy_23train*100,2)}")
print('\n')

"""
Next, the code filters out the first class by creating a new dataframe where the fourth column of 
train_data is not equal to "class-2". This filtered data is then saved to a new file "filtered_train_data_23" using to_csv() function.
"""

# Filter out class 2
filtered_data_13 = train_data[train_data[4] != 'class-2']
filtered_data_13.to_csv("filtered_train_data_13", index = False, header= False)
test_data_filtered_13= test_data[test_data[4] != 'class-2']
test_data_filtered_13.to_csv("filtered_test_data_13", index = False, header= False)

"""
Then, the filtered data is converted to NumPy arrays and the target variable is converted to numerical values using a 
dictionary that maps class labels to numerical values.
"""

traindata = np.array(filtered_data_13)
testdata = np.array(test_data_filtered_13)
target_var_dict_13 = {'class-1': 1, 'class-3': -1}

# Split into features and target variables
X_train, y_train_string = traindata[:, :-1], traindata[:, -1]
y_train_13 = np.array([target_var_dict_13[yi] for yi in y_train_string])

X_test, y_test_string = testdata[:, :-1], testdata[:, -1]
y_test_13 = np.array([target_var_dict_13[yi] for yi in y_test_string])

"""
Next, a perceptron model is instantiated with desired hyperparameters. The learning_rate is set to 1, iterations is set
 to 20, and regularisation_coefficient is set to 0. 
"""

# Instantiate Perceptron with desired hyperparameters
perceptron = Perceptron(learning_rate=1, iterations=20, regularisation_coefficient=0)

"""
The perceptron is then fitted to the training data using the fit() method.
X_train as x and y_train_13 as y in fit method.
"""

# Fit Perceptron to training data
perceptron.fit(X_train, y_train_13)

"""
The target variable is then predicted for the test data and train data using the predict() method.
"""

# Predict target variable for test data
y_pred_13 = perceptron.predict(X_test)
x_pred_13 = perceptron.predict(X_train)

"""
The performance of the model is then evaluated using the mean() and round() functions to 
calculate the accuracy. The accuracy is printed for each pair of classes.
"""

# Evaluate performance
accuracy_13test = np.mean(y_pred_13 == y_test_13)
accuracy_13train = np.mean(x_pred_13 == y_train_13)

print("\033[4mclass-1 vs class-3\033[0m")
print(f"test Accuracy : {round(accuracy_13test*100,2)}")
print(f"train Accuracy: {round(accuracy_13train*100,2)}")
print('\n')

"""
This code block trains and evaluates an OvRPerceptron model for each L2 regularization coefficient in a list.
For each coefficient, the code first splits the data into features and target variables for both training and testing sets.
Then, it trains the OvRPerceptron model with the specified regularisation_coefficient value using the training set and makes predictions on the testing set.
Finally, it calculates and prints the accuracy of the model on both the testing and training sets.
"""
#L2 regularization coefficients to test
l2_coeffs = [0, 0.01, 0.1, 1.0, 10.0, 100.0]

#Iterate through each L2 coefficient
"""
data is converted to NumPy arrays and the target variable is converted to numerical values using a 
dictionary that maps class labels to numerical values.

Next, a ovrperceptron model is instantiated with desired hyperparameters. The learning_rate is set to 1, iterations is set
 to 20, and regularisation_coefficient is set to regularisation_coefficient to get the values from l2_coeffs. 

The perceptron is then fitted to the training data using the fit() method.
The target variable is then predicted for the test data and train data using the predict() method.

The performance of the model is then evaluated using the mean() and round() functions to 
calculate the accuracy. The accuracy is printed for each pair of classes.

"""
print("\033[1m\033[4mONE VS REST CLASSIFICATION\033[0m\033[0m")
print('\n')
for regularisation_coefficient in l2_coeffs:

    traindata = np.array(train_data)
    testdata = np.array(test_data)
    target_var_dict = {'class-1': 0, 'class-2': 1,'class-3': 2}

    # Split into features and target variables
    X_train, y_train_string = traindata[:, :-1], traindata[:, -1]
    y_train = np.array([target_var_dict[yi] for yi in y_train_string])

    X_test, y_test_string = testdata[:, :-1], testdata[:, -1]
    y_test = np.array([target_var_dict[yi] for yi in y_test_string])

    ovrperceptron = OvRPerceptron(learning_rate=1, iterations=20,regularisation_coefficient=regularisation_coefficient)
    ovrperceptron.fit(X_train, y_train)
    y_pred = ovrperceptron.predict(X_test)
    x_pred = ovrperceptron.predict(X_train)
    # Calculate accuracy
    testaccuracy = np.mean(y_pred == y_test)
    print(f"coeff = {regularisation_coefficient}, test accuracy = {round(testaccuracy*100,2)}")
    trainaccuracy = np.mean(x_pred == y_train)
    print(f"coeff = {regularisation_coefficient}, train accuracy = {round(trainaccuracy*100,2)}")
    print('\n')
