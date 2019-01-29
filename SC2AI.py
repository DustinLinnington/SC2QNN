# import sc2
# from sc2 import run_game, maps, Race, Difficulty
# from sc2.constants import *
# from sc2.player import Bot, Computer
# from IBMQuantumExperience import IBMQuantumExperience
# import qiskit
import json
import math

class NeuralNetwork():
	_input_layer = []
	_hidden_layers = []
	_output_layer = []
	_connections = []

	_input_layer_depth = 0
	_hidden_layer_width = 0
	_hidden_layer_depth = 0
	_output_layer_depth = 0

	def __init__(self, input_layer_depth, hidden_layer_width, hidden_layer_depth, output_layer_depth):
		self._input_layer_depth = input_layer_depth
		self._hidden_layer_width = hidden_layer_width
		self._hidden_layer_depth = hidden_layer_depth
		self._output_layer_depth = output_layer_depth

		self._input_layer = []
		self._hidden_layers = []
		self._output_layer = []
		self._connections = []
		self._neurons = []

	def get_training_data(filename):
		training_data_inputs = []
		training_data_outputs = []
		training_data = dict()

		with open(filename, "r") as read_file:
			data = json.load(read_file)

		for dataset in data["Datasets"]:
			training_input_layer = []
			training_output_layer = []
			for input_neuron in data["Datasets"][dataset]["Inputs"]:
				training_input_layer.append(input_neuron)
			for output_neuron in data["Datasets"][dataset]["Outputs"]:
				training_output_layer.append(output_neuron)

			training_data_inputs.append(training_input_layer)
			training_data_outputs.append(training_output_layer)

		training_data["InputData"] = training_data_inputs
		training_data["OutputData"] = training_data_outputs
		return training_data

	def load_neural_net(filename):
		input_layer_depth = 0
		hidden_layer_width = 0
		hidden_layer_depth = 0
		output_layer_depth = 0

		with open(filename, "r") as read_file:
			data = json.load(read_file)

		input_layer_depth = data["Input Layer"]["Layer Depth"]
		hidden_layer_width = data["Hidden Layers"]["Layer Width"]
		hidden_layer_depth = data["Hidden Layers"]["Layer Depth"]
		output_layer_depth = data["Output Layer"]["Layer Depth"]

		loaded_network = NeuralNetwork(input_layer_depth, hidden_layer_width, hidden_layer_depth, output_layer_depth)
		loaded_network.initiate_neural_network(False)

		for connection in data["Connections"]:
			weight = float(data["Connections"][str(connection)]["Weight"])
			new_connection = Connection(loaded_network._neurons[int(data["Connections"][str(connection)]["Connected Neurons"][0])])
			new_connection._weight = weight
			loaded_network._neurons[int(data["Connections"][str(connection)]["Connected Neurons"][1])].add_incoming_connection(new_connection)
			loaded_network._connections.append(new_connection)

		return(loaded_network)

	def save_neural_net(self, filename):
		neural_net_file = open(filename, "w")
		neural_net_file.write("{\n")

		neural_net_file.write("\t\"Input Layer\": {\n")
		neural_net_file.write("\t\t\"Layer Depth\": "  + str(self._input_layer_depth) + ",\n")
		neural_net_file.write("\t\t\"Neuron Layer IDs\": {\n")
		neural_net_file.write("\t\t\t\"0\": [")
		for iteration, neuron in enumerate(self._input_layer):
			neural_net_file.write(str(neuron._id))
			if iteration != len(self._input_layer) - 1:
				neural_net_file.write(", ")
		neural_net_file.write("]\n\t\t}\n\t},\n")

		neural_net_file.write("\t\"Hidden Layers\": {\n")
		neural_net_file.write("\t\t\"Layer Width\": " + str(self._hidden_layer_width) + ",\n")
		neural_net_file.write("\t\t\"Layer Depth\": " + str(self._hidden_layer_depth) + ",\n")
		neural_net_file.write("\t\t\"Neuron Layer IDs\": {\n")
		for layer, hidden_layer in enumerate(self._hidden_layers):
			neural_net_file.write("\t\t\t\"" + str(layer) + "\": [")
			for iteration, neuron in enumerate(self._hidden_layers[layer]):
				neural_net_file.write(str(neuron._id)) 
				if iteration != self._hidden_layer_depth - 1:
					neural_net_file.write(", ")
			if layer != self._hidden_layer_width - 1:
				neural_net_file.write("],\n")
			else:
				neural_net_file.write("]\n")
		neural_net_file.write("\t\t}\n")
		neural_net_file.write("\t},\n")

		neural_net_file.write("\t\"Output Layer\": {\n")
		neural_net_file.write("\t\t\"Layer Depth\": " + str(self._output_layer_depth) + ",\n")
		neural_net_file.write("\t\t\"Neuron Layer IDs\": {\n")
		neural_net_file.write("\t\t\t\"0\": [")
		for iteration, neuron in enumerate(self._output_layer):
			neural_net_file.write(str(neuron._id))
			if iteration != len(self._output_layer) - 1:
				neural_net_file.write(", ")
		neural_net_file.write("]\n\t\t}\n\t},\n")

		neural_net_file.write("\t\"Connections\": {\n")
		for iteration, connection in enumerate(self._connections):
			neural_net_file.write("\t\t\"" + str(iteration) +"\": {\n")
			neural_net_file.write("\t\t\t\"Weight\": " + str(connection._weight) + ",\n")
			neural_net_file.write("\t\t\t\"Connected Neurons\": [" + str(connection._originating_neuron._id))
			neural_net_file.write(", ")
			neural_net_file.write(str(connection._connected_neuron._id))
			if iteration != (len(self._connections) - 1):
				neural_net_file.write("]\n\t\t},\n")
			else:
				neural_net_file.write("]\n\t\t}\n")

		neural_net_file.write("\t}\n")
		neural_net_file.write("}")

	def initiate_neural_network(self, add_connections):
	# Create the neurons for each layer (input, hidden and output)
		for neuron in range(self._input_layer_depth):
			input_neuron = Neuron()
			self._input_layer.append(input_neuron)
			self._neurons.append(input_neuron)

		for layer in range(self._hidden_layer_width):
			hidden_layer = []
			for neuron in range(self._hidden_layer_depth):
				hidden_neuron = Neuron()
				hidden_layer.append(hidden_neuron)
				self._neurons.append(hidden_neuron)
			self._hidden_layers.append(hidden_layer)

		for neuron in range(self._output_layer_depth):
			output_neuron = Neuron()
			self._output_layer.append(output_neuron)
			self._neurons.append(output_neuron)

		if (add_connections):
			# Create the connections for each neuron to every other neuron in the next layer
			for layer, hidden_layer in enumerate(self._hidden_layers):
				for hidden_neuron in self._hidden_layers[layer]:
					if layer == 0:
						for input_neuron in self._input_layer:
							connection = Connection(input_neuron)
							hidden_neuron.add_incoming_connection(connection)
							self._connections.append(connection)
					else:
						for previous_layer_hidden_neuron in self._hidden_layers[layer - 1]:
							connection = Connection(previous_layer_hidden_neuron)
							hidden_neuron.add_incoming_connection(connection)
							self._connections.append(connection)

			for output_neuron in self._output_layer:
				for hidden_neuron in self._hidden_layers[self._hidden_layer_width - 1]:
					connection = Connection(hidden_neuron)
					output_neuron.add_incoming_connection(connection)
					self._connections.append(connection)

	def get_result(self, inputs):
		if (len(inputs) != len(self._input_layer)):
			print ("Get Result input length and input layer mismatch")
			return

		for iteration, neuron in enumerate(self._input_layer):
			if (inputs[iteration] == 1):
				neuron.fire()

		output_values = []
		for neuron in self._output_layer:
			if (neuron._has_fired):
				output_values.append(1)
			else:
				output_values.append(0)
		return output_values

	def calculate_loss(self, expected_output, actual_output):
		if len(expected_output) != len(actual_output):
			print("Output layer size mismatch when calculating loss.")
			return

		output_loss = []
		for neuron, output_value in enumerate(expected_output):
			output_loss.append(math.pow(math.fabs(output_value - actual_output[neuron]), 2))

		return output_loss

	def learn(self, training_filename, learning_rate):
		training_data = self.get_training_data(training_filename)

		# loss = self.calculate_loss(training_data["OutputData"], self._output_layer)

class Neuron():
	def __init__(self):
		self._accumulated_weight = 0
		self._threshold = 1
		self._bias = 0
		self._has_fired = False
		self._outgoing_connections = []
		self._incoming_connections = []
		self._id = Neuron.last_id + 1
		self._connected_neurons_fired = 0

		Neuron.update_id(self)

	def update_id(self):
		Neuron.last_id = Neuron.last_id + 1

	def check_if_fire(self):
		if (self._has_fired):
			return
		accumulated_weight = 0
		for connection in self._incoming_connections:
			if (connection._should_fire == True):
				accumulated_weight += connection._weight
				connection._should_fire = False

		if self.get_sigma(accumulated_weight + self._bias) >= 0.5:
			# TODO: this should fire the neuron... but hopefully in a way that doesn't set a variable
			self.fire()
		
	def fire(self):
		self._has_fired = True
		if (self._outgoing_connections is None):
			return
		for connection in self._outgoing_connections:
			connection.fire()

	def get_sigma(self, _accumulated_weight):
		return 1 / (1 + math.exp(-_accumulated_weight))

	def add_outgoing_connection(self, connection):
		if (connection not in self._outgoing_connections):	
			self._outgoing_connections.append(connection)

	def add_incoming_connection(self, connection):
		if (connection not in self._incoming_connections):
			self._incoming_connections.append(connection)
			connection.add_connected_neuron(self)

	_id = 0
	last_id = -1

class Connection():
	def __init__(self, originating_neuron):
		self._originating_neuron = originating_neuron
		self._weight = 0.5
		self._originating_neuron.add_outgoing_connection(self)
		self._connected_neuron = None
		self._should_fire = False

	def fire(self):
		self._should_fire = True

	# def get_value(self):
	# 	return self._weight * neuron.

	def add_connected_neuron(self, connected_neuron):
		self._connected_neuron = connected_neuron

# neuralNet = NeuralNetwork.load_neural_net("Stuxtnet.txt")
# neuralNet.save_neural_net("Stuxtnet.txt")

training_data = NeuralNetwork.get_training_data("training_data.txt")
input_layer_depth = len(training_data["InputData"][0])
output_layer_depth = len(training_data["OutputData"][0])
neuralNet = NeuralNetwork(input_layer_depth, 3, 5, output_layer_depth)
neuralNet.initiate_neural_network(True)
print(neuralNet.get_result([1, 1, 1, 1, 1, 1, 1]))
for neuron in neuralNet._neurons:
	print(neuron._has_fired)
neuralNet.save_neural_net("Stuxtnet.txt")


# class ZerglingRush(sc2.BotAI):
# 	def __init__(self):
# 		self.drone_counter = 0
# 		self.zergling_counter = 0
# 		self.extractor_started = False
# 		self.spawning_pool_started = False
# 		self.moved_workers_to_gas = False
# 		self.moved_workers_from_gas = False
# 		self.queeen_started = False
# 		self.mboost_started = False

# 	async def on_step(self, iteration):
# 		if iteration == 0:
# 			await self.chat_send("HEY!")

# 		larvae = self.units(LARVA)
# 		zerglings = self.units(ZERGLING)
# 		target = self.enemy_start_locations[0]
# 		hatchery = self.units(HATCHERY).ready.first

# 		for queen in self.units(QUEEN).idle:
# 			abilities = await self.get_available_abilities(queen)
# 			if AbilityId.EFFECT_INJECTLARVA in abilities:
# 				await self.do(queen(EFFECT_INJECTLARVA, hatchery))

# 		if self.supply_left <= 2:
# 			if self.can_afford(OVERLORD) and larvae.exists:
# 				await self.do(larvae.random.train(OVERLORD))
# 		elif self.supply_left > 0 and self.drone_counter < 5:
# 			if self.can_afford(DRONE) and larvae.exists:
# 				await self.do(larvae.random.train(DRONE))
# 				self.drone_counter += 1
# 				print (self.drone_counter)
# 		elif self.units(SPAWNINGPOOL).ready.exists and self.queeen_started == False:
# 			if self.can_afford(QUEEN):
# 				self.queeen_started = True
# 				await self.do(hatchery.train(QUEEN))
# 		elif self.units(SPAWNINGPOOL).ready.exists:
# 			if self.can_afford(ZERGLING) and larvae.exists:
# 				await self.do(larvae.random.train(ZERGLING))
# 				self.zergling_counter += 1

# 		if self.can_afford(SPAWNINGPOOL) and self.spawning_pool_started == False:
# 			pos = hatchery.position.to2.towards(self.game_info.map_center, 6)
# 			drone = self.workers.random
# 			self.spawning_pool_started = True
# 			await self.do(drone.build(SPAWNINGPOOL, pos))

# 		if self.zergling_counter > 10:
# 			for zergling in self.units(ZERGLING).idle:
# 				await self.do(zergling.attack(target))
# 			for worker in self.units(DRONE):
# 				await self.do(worker.attack(target))

# run_game(maps.get("Abyssal Reef LE"), [
# 	Bot(Race.Zerg, ZerglingRush()),
# 	Computer(Race.Protoss, Difficulty.Medium)
# ], realtime=False)