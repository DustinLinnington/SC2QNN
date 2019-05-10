import numpy as np
import json
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
		cost = -np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1-A))) / m
		# Fixes dimensions - turns [[20]] into 20
		cost = np.squeeze(cost)
		return cost

class NeuralNetwork:
	_networkData = 0
	_networkLayers = 0

	def __init__(self, networkLayers):
		self._networkData = 0
		self._networkLayers = networkLayers

	@staticmethod
	def load_network(fileName, networkLayers = 0):
		jsonData = ""
		with open(fileName, "r") as readFile:
			jsonData = json.load(readFile)

		networkLayers = jsonData["NetworkLayers"]
		neuralNet = NeuralNetwork(networkLayers)
		del jsonData["NetworkLayers"]

		for obj in jsonData:
			if isinstance(jsonData[obj], list):
				jsonData[obj] = np.array(jsonData[obj])

		neuralNet._networkData = jsonData
		return neuralNet

	def save_network(self, fileName):
		jsonData = self._networkData
		jsonData["NetworkLayers"] = self._networkLayers
		# Convert all our numpy arrays into lists so it can be JSONified
		for obj in jsonData:
			if isinstance(jsonData[obj], np.ndarray):
				jsonData[obj] = jsonData[obj].tolist()

		with open(fileName, "w") as writeFile:
			json.dump(jsonData, writeFile)
			print("Neural Network data saved to " + fileName)

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
				networkData["Z" + str(l) + ":activation"] = Math.ActivationFunction.SIGMOID.value
			else:
				networkData["A" + str(l)] = Math.calc_A(self._networkData["Z" + str(l)], Math.ActivationFunction.RELU)
				networkData["Z" + str(l) + ":activation"] = Math.ActivationFunction.RELU.value

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
			prevLayerSize = self._networkData["A" + str(l-1)].shape[1]
			gradients["dZ" + str(l)] = Math.calc_A(self._networkData["Z" + str(l)], self._networkData["Z" + str(l) + ":activation"], gradients["dA" + str(l)], True)
			gradients["dW" + str(l)] = np.dot(gradients["dZ" + str(l)], self._networkData["A" + str(l-1)].T) / prevLayerSize
			gradients["db" + str(l)] = np.sum(gradients["dZ" + str(l)], axis = 1, keepdims = True) / prevLayerSize
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
			for l in range(1, len(self._networkLayers)):
				self._networkData["W" + str(l)] = self._networkData["W" + str(l)] - learningRate * self._networkData["dW" + str(l)] 
				self._networkData["b" + str(l)] = self._networkData["b" + str(l)] - learningRate * self._networkData["db" + str(l)]

			if (i % 100 == 0):
				cost = Math.calc_cost(trueOutputData, self._networkData["A" + str(len(self._networkLayers) - 1)])
				print("Cost at Iteration " +  str(i) + ":", cost)

	def predict(self, inputData):
		self._networkData["A0"] = inputData
		networkData = self.forward_propagate()
		predictedOutput = networkData["A" + str(len(self._networkLayers) - 1)]
		np.squeeze(predictedOutput)
		return predictedOutput