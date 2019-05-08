import numpy as np
from enum import Enum

class Math:
	class ActivationFunction(Enum):
		RELU = 1
		LEAKY_RELU = 2
		SIGMOID = 3
		TANH = 4

	@staticmethod
	def calc_Z(weights, bias, prevActivation):
		Z = weights.dot(prevActivation) + bias
		return Z

	@staticmethod
	def calc_A(Z, activationFunction, dA = 0, takeDerivative = False):
		A = Z
		if (activationFunction == Math.ActivationFunction.RELU):
			A = Math.calc_RELU(Z, dA, takeDerivative)
		elif(activationFunction == Math.ActivationFunction.SIGMOID):
			A = Math.calc_sigmoid(Z, dA, takeDerivative)
		elif(activationFunction == Math.ActivationFunction.TANH):
			A = Math.calc_tanH(Z, dA, takeDerivative)
		return A

	@staticmethod
	def calc_sigmoid(Z, dA, takeDerivative = False):
		sigmoid = 1/(1 + np.exp(-Z))
		if (takeDerivative == True):
			return dA * sigmoid * (1 - sigmoid)
		else:
			return sigmoid

	@staticmethod
	def calc_tanH(Z, dA, takeDerivative = False):
		tanh = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
		if (takeDerivative == True):
			return dA * (1 - tanh ** 2)
		else:
			return tanh

	@staticmethod
	def calc_RELU(Z, dA, takeDerivative = False):
		relu = np.maximum(0, Z)
		if (takeDerivative == True):
			# Ensure dZ is a correct object
			dZ = np.array(dA, copy = True)
			# When Z <= 0, dZ should also be 0
			dZ[Z <= 0] = 0
			return(dZ)
		else:
			return relu


	@staticmethod
	def calc_cost(trueOutputData, calculatedOutputData):
		m = trueOutputData.shape[1]
		Y = trueOutputData
		A = calculatedOutputData
		# cost = (1./m) * (-np.dot(Y,np.log(A).T) - np.dot(1-Y, np.log(1-A).T))

		cost = -np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1-A))) / m
		# Fixes dimensions - turns [[20]] into 20
		cost = np.squeeze(cost)
		return cost

class NeuralNetwork:
	_networkData = 0
	_networkLayers = 0

	def __init__(self, networkLayers, fileName = 0):
		self._parameters = 0
		if (fileName != 0):
			self.load_network(fileName)
		else:
			self._networkLayers = networkLayers

	def load_network(self, fileName):
		pass

	def save_network(self, fileName):
		pass

	def initialize_network(self):
		"""
    	Arguments:
    	_networkLayers -- python array (list) containing the dimensions of each layer in our network
    
    	Returns:
    	_networkData -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        	Wl -- weight matrix of shape (_networkLayers[l], _networkLayers[l-1])
        	bl -- bias vector of shape (_networkLayers[l], 1)
    	"""
		np.random.seed(1)

		_networkData = {}
		for l in range(1, len(self._networkLayers)):
			_networkData["W" + str(l)] = np.random.randn(self._networkLayers[l], self._networkLayers[l - 1]) / np.sqrt(self._networkLayers[l-1])
			_networkData["b" + str(l)] = np.zeros((self._networkLayers[l], 1))

			assert(_networkData['W' + str(l)].shape == (self._networkLayers[l], self._networkLayers[l-1]))
			assert(_networkData['b' + str(l)].shape == (self._networkLayers[l], 1))

		return _networkData

	def forward_propagate(self):
		networkData = self._networkData

		for l in range(1, len(self._networkLayers)):
			networkData["Z" + str(l)] = Math.calc_Z(self._networkData["W" + str(l)], self._networkData["b" + str(l)], self._networkData["A" + str(l - 1)])

			# Activation function of the last layer is sigmoid so we can get a value from 0 to 1
			if (l == len(self._networkLayers) - 1):
				networkData["A" + str(l)] = Math.calc_A(self._networkData["Z" + str(l)], Math.ActivationFunction.SIGMOID)
				networkData["Z" + str(l) + ":activation"] = Math.ActivationFunction.SIGMOID
			else:
				networkData["A" + str(l)] = Math.calc_A(self._networkData["Z" + str(l)], Math.ActivationFunction.RELU)
				networkData["Z" + str(l) + ":activation"] = Math.ActivationFunction.RELU

		return networkData

	# Calculate gradients for each layer of the Neural Net by using derivatives
	def backward_propagate(self, trueOutputData):
		numberOfTrainingExamples = trueOutputData.shape[0]

		# Calculate derivatives from the end of the neural net to the beginning, using chain rule
		gradients = {}
		m = len(self._networkLayers)
		# Take the derivitave of the loss function (d of a of the last layer)
		AL = self._networkData["A" + str(m - 1)]
		gradients["dA" + str(m - 1)] = -(np.divide(trueOutputData, AL) - np.divide(1 - trueOutputData, 1 - AL))
		# Go through the layers one-by-one starting from the second last layer
		for l in reversed(range(1, m)):
			gradients["dZ" + str(l)] = Math.calc_A(self._networkData["Z" + str(l)], self._networkData["Z" + str(l) + ":activation"], gradients["dA" + str(l)], True)
			gradients["dW" + str(l)] = np.dot(gradients["dZ" + str(l)], self._networkData["A" + str(l-1)].T) / m
			gradients["db" + str(l)] = np.sum(gradients["dZ" + str(l)], axis = 1, keepdims = True) / m
			gradients["dA" + str(l - 1)] = np.dot(self._networkData["W" + str(l)].T, gradients["dZ" + str(l)])

		return gradients

	def train_network(self, inputData, trueOutputData, learningRate, iterations):
		# If network has never been ran before, initialize weight and bias vectors to random/zero values
		if (self._networkData == 0):
			self._networkData = self.initialize_network()
		self._networkData["A0"] = inputData

		for i in range(iterations):
			self._networkData = self.forward_propagate()
			self._networkData.update(self.backward_propagate(trueOutputData))
			# print(self._networkData)
			for l in range(1, len(self._networkLayers)):
				print ("W Layer:" + str(l))
				print(self._networkData["W" + str(l)])
				print ("dZ Layer:" + str(l))
				print(self._networkData["dZ" + str(l)])
				self._networkData["W" + str(l)] = self._networkData["W" + str(l)] - learningRate * self._networkData["dW" + str(l)] 
				self._networkData["b" + str(l)] = self._networkData["b" + str(l)] - learningRate * self._networkData["db" + str(l)]
				print(self._networkData["W" + str(l)])

			# if (i % 100 == 0):
			cost = Math.calc_cost(trueOutputData, self._networkData["A" + str(len(self._networkLayers) - 1)])
			print("Cost at Iteration " +  str(i) + ":", cost)

	def predict(self, inputData):
		self._networkData["A0"] = inputData
		networkData = self.forward_propagate()
		predictedOutput = networkData["A" + str(len(self._networkLayers) - 1)]
		predictedOutput = (predictedOutput > 0.5)
		np.squeeze(predictedOutput)
		return predictedOutput

# inputData = np.array([[1, 0, 1, 4, 5, 0, 6, 9, 1, 0], 
# 					[4, 0, 2, 2, 3, 7, 0, 1, 4, 8], 
# 					[6, 1, 5, 6, 5, 6, 3, 0, 1, 0], 
# 					[0, 1, 0, 3, 4, 7, 6, 6, 6, 4], 
# 					[8, 0, 7, 4, 3, 7, 0, 0, 1, 4]])
np.random.seed(1)
inputData = np.random.randn(1, 1) * 0.001
outputData = np.sin(inputData)
networkLayers = np.array([1, 2, 2, 2, 1])

neuralNet = NeuralNetwork(networkLayers)
neuralNet.train_network(inputData, outputData, 0.0075, 1)
# print("Prediction 1: ", neuralNet.predict(np.array([[25.904]])))

# # neuralNet.train_network(inputData, outputData, 0.0075, 2000)
# # print("Prediction 2: ", neuralNet.predict(np.array([[25.904]])))

# print("Actual Result Should Be: " + str(np.sin(25.904)))
























# import numpy as np
# from enum import Enum

# class Math():
# 	# m = Number of training examples
# 	# X = All training sets of our input data in a matrix
# 	# Z = Calculated output before being tuned up by activation function
# 	# A = Calculated output after activation function
# 	# Y = What the results should have been
# 	# gPrime = Derivitave of the slope of some value
# 	# cost = A value based on how close the predicted outcome vs the real outcome was

# 	class ActivationFunction(Enum):
# 		RELU = 1
# 		LEAKY_RELU = 2
# 		SIGMOID = 3
# 		TANH = 4

# 	@staticmethod
# 	def calcRegression(inputData, weights, bias):
# 		return np.dot(weights.T, inputData) + bias

# 	@staticmethod
# 	def calcActivation(activationFunction, inputData, weights, bias):
# 		Z = Math.calcRegression(inputData, weights, bias)
# 		A = 0
# 		if(activationFunction == Math.ActivationFunction.RELU):
# 			A = np.maximum(0, Z)
# 		elif(activationFunction == Math.ActivationFunction.LEAKY_RELU):
# 			A = np.maximum(0.01 * Z, Z)
# 		elif(activationFunction == Math.ActivationFunction.SIGMOID):
# 			A = Math.calcSigmoid(Z)
# 		elif(activationFunction == Math.ActivationFunction.TANH):
# 			A = Math.calcTanH(Z)
# 		return A

# 	@staticmethod
# 	def calcSigmoid(Z):
# 		return 1/(1 + np.exp(-Z))

# 	def calcTanH(Z):
# 		return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

# 	@staticmethod
# 	def calcCost(activationFunction, A, Y, m):
# 		cost = 0
# 		if (activationFunction == Math.ActivationFunction.RELU):
# 			gPrime = 0 if A <= 0 else 1
# 		elif (activationFunction == Math.ActivationFunction.LEAKY_RELU):
# 			gPrime = 0.01 if A <= 0 else 1
# 		elif (activationFunction == Math.ActivationFunction.SIGMOID):
# 			cost = -((np.sum(Y * np.log(A) + ((1 - Y) * np.log(1-A)))) / m)
# 		return cost

# 	@staticmethod
# 	def calcGradients(inputData, trueOutput, weights, bias, activationFunction):
# 		m = inputData.shape[1]
# 	    # FORWARD PROPAGATION
# 		A = Math.calcActivation(activationFunction, inputData, weights, bias)
# 		cost = Math.calcCost(activationFunction, A, trueOutput, m)
	    
# 	    # BACKWARD PROPAGATION
# 		dw = np.dot(inputData, (A - trueOutput).T) / m
# 		db = np.sum(A - trueOutput) / m

# 		assert(dw.shape == weights.shape)
# 		assert(db.dtype == float)
# 		cost = np.squeeze(cost)
# 		assert(cost.shape == ())
	    
# 		gradients = {"dw": dw,
# 	             "db": db}
	    
# 		return gradients, cost

# class NeuralNetwork():
# 	_weights = []
# 	_bias = []
# 	def __init__(self):
# 		self._weights = []
# 		self._bias = []

# 	class TrainingData():
# 		def __init__(trainingFile):
# 			pass

# 	def createNetwork(self, inputData):
# 		self._weights = np.random.rand(inputData.shape[0], 1) * 0.01
# 		self._bias = 0

# 	def trainNetwork(self, inputData, trueOutput, iterations, learningRate):
# 		costs = []
    
# 		for i in range(iterations):
# 			# Cost and gradient calculation
# 			gradients, cost = Math.calcGradients(inputData, trueOutput, self._weights, self._bias, Math.ActivationFunction.SIGMOID)
        
# 			# Retrieve derivatives from gradients
# 			dw = gradients["dw"]
# 			db = gradients["db"]
        
# 			# Update weights and biases
# 			self._weights = self._weights - learningRate * dw
# 			self._bias = self._bias - learningRate * db
        
# 			# # Record the costs
# 			# if i % 1000 == 0:
# 			# 	costs.append(cost)
        
# 			# # Print the cost every 100 training iterations
# 			# if i % 10 == 0:
# 			# 	print ("Cost after iteration %i: %f" %(i, cost))

# 	def predict(self, inputData):
# 		A = Math.calcActivation(Math.ActivationFunction.RELU, inputData, self._weights, self._bias)

# 		print(A)


# # A = np.random.randn(4,3)
# # B = np.sum(A, axis = 1, keepdims = True)
# # print(B)

# inputData = np.array([[0., 1., 1.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
# print(inputData.shape)
# # trueOutput = np.array([0., 1., 0.])
# # neuralNet = NeuralNetwork()
# # neuralNet.createNetwork(inputData)
# # neuralNet.trainNetwork(inputData, trueOutput, 10000, 0.0009)
# # testData = np.array([[1., 1.], [1., 0.], [0., 0.], [1., 1.]])
# # neuralNet.predict(testData)

