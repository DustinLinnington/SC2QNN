/************ NEURAL NETWORK ************\
-         Created Aug 24, 2019	    	 -
-         Created By: Dustin Daon        -
-     Contact: Dustin.Daon@gmail.com     -
------------------------------------------

USE:
=================================================================
Create a new NeuralNetwork object and give it the parameter of an
array describing the layers in the network and the amount of 
neurons per layer. Ex: A network with 3 hidden layers containing 
4 neurons per layer, 1 output layer with 1 neuron and 5 inputs
would look like: [5, 4, 4, 4, 1]

To train it, it only requires that you give it:
1) Input Array: a numpy array of n x m dimensions where n = number of inputs and 
m = training examples,
2) Output Array: a numpy array containing the results of the training examples
(with a shape of [output neurons, training examples])
3) Hyperparameter: a learing rate. The higher the number the 'faster' it learns,
but too high of a number and it can overshoot the correctness of
its predictions which would result in a poorly trained network
(a good value for this would be 0.0075 - 0.09)
4) Hyperparameter: number of training iterations. The higher the number, the better
it will learn your training set. However, this will take more
time and you run the risk of it overfitting to the training set if
this number is too high.

After training, you can save the neural network to a JSON file
to be loaded again in the future. You can also predict a result
from a trained network by feeding it some input data.

=================================================================
EXAMPLE
=================================================================

# Array of 3 hidden layers (size 3, 2, 2), 1 output layer (size 1)
# and 5 input values
networkLayers = numpy.array([5, 3, 2, 2, 1])

# Input data is 10 training examples with 5 values each
inputData = numpy.array([[1, 0, 1, 4, 5, 0, 6, 9, 1, 0], 
 					[4, 0, 2, 2, 3, 7, 0, 1, 4, 8], 
 					[6, 1, 5, 6, 5, 6, 3, 0, 1, 0], 
 					[0, 1, 0, 3, 4, 7, 6, 6, 6, 4], 
 					[8, 0, 7, 4, 3, 7, 0, 0, 1, 4]])

# Output data is the correct result of the input data with shape 
# of 1x10 (size of output layer x number of training examples)
outputData = numpy.array([[1, 0, 0, 1, 0, 0, 0, 1, 1, 1]])

# Create a new neural network of shape networkLayers
neuralNet = NeuralNetwork(networkLayers)

# Train our new neural network 1000 times with a learning rate
# of 0.0075
neuralNet.train_network(inputData, outputData, 0.0075, 1000)

# Predict an outcome from our trained network using 1 test example
testData = numpy.array([[1], [6], [8], [8], [9]])
prediction = neuralNet.predict(testData)
print(prediction)

# Save our trained network
neuralNet.save_network("neural_net.txt")

# Load a different network
starcraftNet = neuralNet.load_network("sc_neural_net.txt")