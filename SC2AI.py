import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer

class Connection:
	pass
class Neuron():
	def __init__(self, connections):
		self._connections = connections
	def fire(self):
		if (self._has_fired == True):
			return
		if (self._connections is None):
			return
		self._has_fired = True
		# for __connection in self._connections:
		self._connections.fire()

	def add_weight(self, weight):
		self._accumulated_weight += weight
		if (weight >= 50):
			self.fire()

	_accumulated_weight = 0
	_has_fired = False
	_connections = []

class Connection():
	def __init__(self, connected_neuron):
		self._connected_neuron = connected_neuron  
	def fire(self):
		self._connected_neuron.add_weight(self._weight)
	_weight = 0.5
	_connected_neuron = Neuron(None)

output_neuron = Neuron(None)
new_connection = Connection(output_neuron)

connections = []
connections.append(new_connection)

input_neuron_connections = connections
input_neuron = Neuron(new_connection)

input_neuron.fire()

print ("Output neuron has a", output_neuron._accumulated_weight * 100, "% chance of being true.")




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