import pandas as pd
import torch
import numpy as np

# Tensor Basics
# import the dataset using pandas
apartments_df = pd.read_csv("streeteasy.csv")

# select the rent, size, and age columns
apartments_df = apartments_df[["rent", "size_sqft", "building_age_yrs"]]

## YOUR SOLUTION HERE ##
apartment = [2000,500,7]
apartment_tensor = torch.tensor(apartment,dtype=torch.int)
# show output
print(apartment_tensor)

apartments_tensor = torch.tensor(apartments_df.values,dtype = torch.float32)
print(apartments_tensor)


#predict rent --> using Linear regression
'''This is an example of a regression problem in machine learning: a problem where we are trying to predict a target numeric value.

In any regression model, we start with certain input features. Using those input features, we try to predict the target or output (sometimes called a label, although this is more common in classification problems).

For example, we might try to build a model with

input feature: square footage of an apartment
target: predicted cost to rent the apartment'''


# Consider this linear equation rent=2.5sqft−1.5age+1000
# Our first step toward building neural networks is to transform this equation into a neural network structure called a Perceptron:
# One of the ways neural networks move beyond linear regression is by incorporating non-linear activation functions. 

''' 
One of the most common 
activation functions
Preview: Docs Activation functions are mathematical functions that introduce non-linearity into the model, enabling neural networks to learn complex patterns from data.
used in neural networks is called ReLU.
If a number is negative, ReLU((Rectified Linear Unit)) returns 0. If a number is positive, ReLU returns the number with no changes.
Activation functions take the linear equation as input, and modify it to introduce non-linearity.

The process with an activation function is

receive the weighted inputs
add them up (to produce the same linear equation as before)
apply an activation function

 '''
 
ReLU(-1)
# output: 0, since -1 is a negative number

ReLU(.5)
# output: .5, since .5 is not negative

'''
 Build a Sequential Neural Network
Now that we know the basic structure of a neural network, let’s build one in PyTorch using PyTorch’s Sequential container
'''
