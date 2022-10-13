import random
from argparse import Namespace
from enum import Enum

import numpy as np
from game import Game
from gym import Env
from gym.spaces import Box, MultiDiscrete
from stable_baselines3 import A2C

from utilities import *


class UNIT_TAGS(Enum):
    TRUCK = "Truck"
    LIGHT_TANK = "LightTank"
    HEAVY_TANK = "HeavyTank"
    DRONE = "Drone"


UNIT_TAGS_BY_INDEX = {i: a.name for i, a in enumerate(UNIT_TAGS)}


class TERRAIN_TAGS(Enum):
    GRASS = 0
    DIRT = 1
    MOUNTAIN = 2
    WATER = 3


TERRAIN_TAGS_BY_INDEX = {i: a.name for i, a in enumerate(TERRAIN_TAGS)}


DIRECTION_OFFSETS_EVEN: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
}
DIRECTION_OFFSETS_ODD: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (-1, -1),
    2: (0, -1),
    3: (1, -1),
    4: (1, 0),
    5: (0, 1),
    6: (-1, 0),
}
DIRECTION_OFFSETS = {
    0: DIRECTION_OFFSETS_EVEN,
    1: DIRECTION_OFFSETS_ODD,
}

ACTION_LENGTH: int = 7


def get_movement_offsets(location: tuple[int, int]) -> tuple[int, int]:
    return DIRECTION_OFFSETS[location[1] % 2]


def calculate_location_by_offset(offset: int, location: tuple[int, int]) -> tuple[int, int]:
    y, x = location
    x_offset, y_offset = offset
    return (y + y_offset, x + x_offset)


class Unit:
    _cost: int = -1
    _attack: int = -1
    _max_hp: int = -1
    _heavy: bool = False
    _fly: bool = False
    _anti_air: bool = False

    _can_move_on = {
        TERRAIN_TAGS.GRASS: False,
        TERRAIN_TAGS.DIRT: False,
        TERRAIN_TAGS.MOUNTAIN: False,
        TERRAIN_TAGS.WATER: False
    }

    will_move = False

    hp: int = -1
    load: int = -1
    location: tuple[int, int] = (-1, -1)
    tag: UNIT_TAGS = None

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: UNIT_TAGS) -> None:
        self.hp = hp
        self.load = load
        self.location = location
        self.tag = tag

    def get_available_directions(self, world) -> list[int]:
        available_directions: list[int] = []
        all_units = world.blue_team.units + world.red_team.units
        height, width = world.terrain.shape
        for direction, offset in get_movement_offsets(self.location).items():
            Y, X = calculate_location_by_offset(offset, self.location)
            coordinate = (Y, X)
            if not (0 <= Y < height and 0 <= X < width):
                continue

            if coordinate in [*world.reserved_locations,
                              world.blue_team.base.location,
                              world.red_team.base.location,
                              *[unit.location for unit in all_units if not unit.will_move]]:
                continue

            if not self._can_move_on[TERRAIN_TAGS(world.terrain[Y][X])]:
                continue

            available_directions.append(direction)

        return available_directions


class HeavyTankUnit(Unit):
    _cost: int = 2
    _attack: int = 2
    _max_hp: int = 4
    _heavy: bool = True
    _fly: bool = False
    _anti_air: bool = False

    _can_move_on = {
        TERRAIN_TAGS.GRASS: True,
        TERRAIN_TAGS.DIRT: False,
        TERRAIN_TAGS.MOUNTAIN: False,
        TERRAIN_TAGS.WATER: False
    }


class LightTankUnit(Unit):
    _cost: int = 1
    _attack: int = 2
    _max_hp: int = 2
    _heavy: bool = False
    _fly: bool = False
    _anti_air: bool = True

    _can_move_on = {
        TERRAIN_TAGS.GRASS: True,
        TERRAIN_TAGS.DIRT: True,
        TERRAIN_TAGS.MOUNTAIN: False,
        TERRAIN_TAGS.WATER: False
    }


class TruckUnit(Unit):
    _cost: int = 1
    _max_hp: int = 1
    _max_load: int = 3
    _heavy: bool = False
    _fly: bool = False
    _anti_air: bool = False

    _can_move_on = {
        TERRAIN_TAGS.GRASS: True,
        TERRAIN_TAGS.DIRT: True,
        TERRAIN_TAGS.MOUNTAIN: False,
        TERRAIN_TAGS.WATER: False
    }

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: UNIT_TAGS) -> None:
        super().__init__(hp, load, location, tag)

    def has_space(self) -> bool:
        return self.load < self._max_load

    def has_load(self) -> bool:
        return 0 < self.load

    def is_on_resource(self, world) -> bool:
        for resource in world.resources:
            if self.location == resource:
                return True

        return False

    def is_on_base(self, world) -> bool:
        return self.location == world.blue_team.base.location

    def get_nearest_resources(self, resources: tuple[int, int]) -> list[tuple[int, int]]:
        pass

    def get_available_directions(self, world) -> list[int]:
        available_directions = super().get_available_directions(world)
        if 0 < self.load:
            for direction, offset in get_movement_offsets(self.location).items():
                Y, X = calculate_location_by_offset(offset, self.location)
                coordinate = (Y, X)
                if coordinate == world.blue_team.base.location:
                    available_directions.append(direction)
                    break

        return available_directions


class DroneUnit(Unit):
    _cost: int = 1
    _attack: int = 1
    _max_hp: int = 1
    _heavy: bool = False
    _fly: bool = True
    _anti_air: bool = True

    _can_move_on = {
        TERRAIN_TAGS.GRASS: True,
        TERRAIN_TAGS.DIRT: True,
        TERRAIN_TAGS.MOUNTAIN: True,
        TERRAIN_TAGS.WATER: True
    }


class Base:
    location: tuple[int, int] = (-1, -1)
    load: int = -1


class Team:
    index: int
    units: list[Unit]
    base: Base

    def __init__(self, index: int) -> None:
        self.index = index
        self.units = []
        self.base = Base()


class World:
    __blue_team: Team
    __red_team: Team

    def __init__(self, blue_team_index: int, red_team_index: int) -> None:
        self.__blue_team = Team(blue_team_index)
        self.__red_team = Team(red_team_index)

    @property
    def blue_team(self) -> Team:
        return self.__blue_team

    @property
    def red_team(self) -> Team:
        return self.__red_team

    height: int = -1
    width: int = -1

    terrain: np.ndarray
    resources: np.ndarray

    reserved_locations: list[dict[int, int]] = []

    def get_size(self) -> tuple[int, int]:
        return self.height, self.width

    def clear(self):
        self.blue_team.units.clear()
        self.red_team.units.clear()
        self.reserved_locations.clear()

    def get_terrain_tag(self, location: tuple[int, int]) -> UNIT_TAGS:
        Y, X = location
        return TERRAIN_TAGS_BY_INDEX[self.terrain[X][Y]]

    def is_there_resource_on_location(self, location: tuple[int, int]):
        for resource in self.resources:
            if resource == location:
                return True

        return False

    def get_distance_between_two_locations(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        if source == None or target == None:
            return -1

        source = list(copy.copy(source))
        target = list(copy.copy(target))
        source[0] -= (source[1] + 1) // 2
        target[0] -= (target[1] + 1) // 2
        return (abs(source[0] - target[0]) + abs(source[1] - target[1]) + abs(source[0] + source[1] - target[0] - target[1])) // 2


class BaseAgent():
    # Private:
    __state: dict

    # Protected:
    _world: World

    def _get_unit(self, entity: list[any]) -> Unit:
        [_, tag, hp, location, load] = entity.values()
        if tag == UNIT_TAGS.HEAVY_TANK.value:
            unit = HeavyTankUnit(hp, load, location, UNIT_TAGS.HEAVY_TANK)
        elif tag == UNIT_TAGS.LIGHT_TANK.value:
            unit = LightTankUnit(hp, load, location, UNIT_TAGS.LIGHT_TANK)
        elif tag == UNIT_TAGS.TRUCK.value:
            unit = TruckUnit(hp, load, location, UNIT_TAGS.TRUCK)
        elif tag == UNIT_TAGS.DRONE.value:
            unit = DroneUnit(hp, load, location, UNIT_TAGS.DRONE)
        else:
            unit = Unit(hp, load, location, UNIT_TAGS(tag))

        return unit

    def _decode_observation(self, observation: dict[str, any]):
        units = observation['units']
        hps = observation['hps']
        bases = observation['bases']
        res = observation['resources']
        load = observation['loads']

        blue = 0
        red = 1
        y_max, x_max = res.shape
        blue_units = []
        red_units = []
        resources = []
        blue_base = None
        red_base = None
        for i in range(y_max):
            for j in range(x_max):
                if units[blue][i][j] < 6 and units[blue][i][j] != 0 and hps[blue][i][j] > 0:
                    blue_units.append(
                        {
                            'unit': units[blue][i][j],
                            'tag': tagToString[units[blue][i][j]],
                            'hp': hps[blue][i][j],
                            'location': (i, j),
                            'load': load[blue][i][j]
                        }
                    )
                if units[red][i][j] < 6 and units[red][i][j] != 0 and hps[red][i][j] > 0:
                    red_units.append(
                        {
                            'unit': units[red][i][j],
                            'tag': tagToString[units[red][i][j]],
                            'hp': hps[red][i][j],
                            'location': (i, j),
                            'load': load[red][i][j]
                        }
                    )
                if res[i][j] == 1:
                    resources.append((i, j))
                if bases[blue][i][j]:
                    blue_base = (i, j)
                if bases[red][i][j]:
                    red_base = (i, j)
        return [blue_units, red_units, blue_base, red_base, resources]

    def _action(self, chunks: list[tuple[int, int]], recruitment: int):
        locations, movements, targets, = [], [], []
        blue_team = self._world.blue_team
        red_team = self._world.red_team
        lower_bound = min(len(blue_team.units), len(chunks))
        for i in range(lower_bound):
            direction, target = None, None
            blue_unit = blue_team.units[i]
            sorted_red_units = sorted(red_team.units, key=lambda unit: self._world.get_distance_between_two_locations(
                blue_unit.location, unit.location), reverse=True)
            chunk = chunks[i]
            available_directions = blue_unit.get_available_directions(
                self._world)
            if len(available_directions) == 0:
                continue

            movement_offsets = get_movement_offsets(blue_unit.location)
            direction, target_index = chunk
            if direction != 0 and direction not in available_directions:
                continue

            if target_index < len(sorted_red_units):
                target = sorted_red_units[target_index].location
            elif 0 < len(sorted_red_units):
                target = sorted_red_units[-1].location

            coordinate = calculate_location_by_offset(
                movement_offsets[direction], blue_unit.location)
            if 0 < direction:
                self._world.reserved_locations.append(coordinate)
            else:
                if isinstance(blue_unit, TruckUnit):
                    if (blue_unit.has_space() and blue_unit.is_on_resource(self._world)) or (blue_unit.has_load() and blue_unit.is_on_base(self._world)):
                        target = blue_unit.location

            distance_to_target = self._world.get_distance_between_two_locations(
                blue_unit.location, target)

            if not isinstance(blue_unit, TruckUnit):
                if distance_to_target < 2:
                    direction = 0

            if None in [direction, target]:
                continue

            blue_unit.will_move = True
            locations.append(blue_unit.location)
            movements.append(direction)
            targets.append(target)

        assert all(len(x) <= self.action_length for x in [
                   locations, movements, targets])
        return (locations, movements, targets, recruitment)

    # Public:
    action_length: int = ACTION_LENGTH

    def __init__(self):
        pass

    def _set_ready_to_action(self, observation: dict[str, any]) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]], int]:
        self._world.clear()

        self._world.terrain = observation["terrain"]
        decoded_state = self._decode_observation(observation)
        scores = observation["score"]

        # Blue Team
        blue_team = self._world.blue_team
        blue_team.base.load = int(scores[blue_team.index])
        blue_team.base.location = decoded_state[blue_team.index + 2]
        for i, entity in enumerate(decoded_state[blue_team.index]):
            unit = self._get_unit(entity)
            blue_team.units.append(unit)

        # Red Team
        red_team = self._world.red_team
        red_team.base.load = int(scores[red_team.index])
        red_team.base.location = decoded_state[red_team.index + 2]
        for i, entity in enumerate(decoded_state[red_team.index]):
            unit = self._get_unit(entity)
            red_team.units.append(unit)

        self._world.resources = decoded_state[4]


class TrainAgent(BaseAgent, Env):
    # Private:
    __state: dict[str, any]
    __episodes: int = 0
    __steps: int = 0
    __reward: float = 0.0

    def __flat_state(self, state: dict[str, any]):
        turn = state['turn']  # 1
        max_turn = state['max_turn']  # 1
        units = state['units']
        hps = state['hps']
        bases = state['bases']
        score = state['score']  # 2
        res = state['resources']
        load = state['loads']
        terrain = state["terrain"]
        y_max, x_max = res.shape
        my_units = []
        enemy_units = []
        resources = []
        for i in range(y_max):
            for j in range(x_max):
                if units[0][i][j] < 6 and units[0][i][j] != 0:
                    my_units.append(
                        {
                            'unit': units[0][i][j],
                            'tag': UNIT_TAGS_BY_INDEX[units[0][i][j] - 1],
                            'hp': hps[0][i][j],
                            'location': (i, j),
                            'load': load[0][i][j]
                        }
                    )
                if units[1][i][j] < 6 and units[1][i][j] != 0:
                    enemy_units.append(
                        {
                            'unit': units[1][i][j],
                            'tag': UNIT_TAGS_BY_INDEX[units[1][i][j] - 1],
                            'hp': hps[1][i][j],
                            'location': (i, j),
                            'load': load[1][i][j]
                        }
                    )
                if res[i][j] == 1:
                    resources.append((i, j))
                if bases[0][i][j]:
                    my_base = (i, j)
                if bases[1][i][j]:
                    enemy_base = (i, j)

        unitss = [*units[0].reshape(-1).tolist(),
                  *units[1].reshape(-1).tolist()]
        hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        basess = [*bases[0].reshape(-1).tolist(),
                  *bases[1].reshape(-1).tolist()]
        ress = [*res.reshape(-1).tolist()]
        loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]
        terr = [*terrain.reshape(-1).tolist()]

        state = (*score.tolist(), turn, max_turn, *unitss,
                 *hpss, *basess, *ress, *loads, *terr)

        return np.array(state, dtype=np.int16)

    # Public:
    observation_space: Box
    action_space: MultiDiscrete

    def __init__(self, kwargs: Namespace, agents: list[str]) -> None:
        super().__init__()
        self.__game = Game(kwargs, agents)
        self._world = World(0, 1)
        self.observation_space = Box(
            low=-2,
            high=401,
            shape=(self.__game.map_x*self.__game.map_y*10+4,),
            dtype=np.int16
        )
        self.action_space = MultiDiscrete(
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5])

    def setup(self, observation_space: Box, action_space: MultiDiscrete):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        self.__episodes += 1
        self.__steps = 0
        self.__state = self.__game.reset()
        self.__reward = 0.0
        return self.__flat_state(self.__state)

    def step(self, sample: np.ndarray):
        movements = sample[0:7]
        targets = sample[7:14]
        recruitment = sample[14]
        chunks = list(zip(movements, targets))
        self._set_ready_to_action(self.__state)
        action = self._action(chunks, recruitment)
        self.__state, _, done = self.__game.step(action)
        self.__steps += 1

        # self.__reward += 1
        reward = self.__reward
        reward += self._world.blue_team.base.load * 16
        reward += sum([unit.load for unit in self._world.blue_team.units if isinstance(unit, TruckUnit)])
        reward -= len([unit for unit in self._world.blue_team.units if not unit.will_move]) * 2
        if len(self._world.red_team.units) == 0:
            reward += 1024
        return self.__flat_state(self.__state), reward, done, {}

    def render(self,):
        return None

    def close(self,):
        return None


# class EvaluationAgent(BaseAgent):
#     # Private:
#     __a2c: A2C

#     # Public:

#     def __init__(self, index: int, action_length: int = ACTION_LENGTH):
#         super().__init__()
#         self.action_length = action_length
#         self._world = World(index, 1 - index)
#         self.__a2c = A2C.load('\\'.join(__file__.split(
#             "\\")[:-1]) + "\\..\\..\\models\\singletruckst\\tsts_192000_steps")

#     def action(self, observation: dict[str, any]):
#         self._set_ready_to_action(observation)
#         return ([], [], [], 0)


class EvaluationAgent(BaseAgent):
    """An example agent which shares the same learning infrastructure as
    the full-fledged benchmarks, but implements random action selection."""

    # Private:
    __a2c: A2C

    # Public:
    observation_space: Box
    action_space: MultiDiscrete

    def __init__(self, observation_space: Box, action_space: MultiDiscrete):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self._world = World(0, 1)
        self.__a2c = A2C.load('\\'.join(__file__.split(
            "\\")[:-1]) + "\\..\\..\\models\\singletruckst\\tsts_192000_steps")

    def action(self, chunks: list[tuple[int, int]], recruitment: int):
        locations, movements, targets, = [], [], []
        blue_team = self._world.blue_team
        lower_bound = min(len(blue_team.units), len(chunks))
        for i in range(lower_bound):
            blue_unit = blue_team.units[i]
            chunk = chunks[i]
            available_directions = blue_unit.get_available_directions(
                self._world)
            if len(available_directions) == 0:
                continue

            movement_offsets = get_movement_offsets(blue_unit.location)
            direction, target = chunk
            if direction not in available_directions:
                continue

            target = calculate_location_by_offset(
                movement_offsets[target], blue_unit.location)
            coordinate = calculate_location_by_offset(
                movement_offsets[direction], blue_unit.location)
            if 0 < direction:
                self._world.reserved_locations.append(coordinate)

            blue_unit.will_move = True
            locations.append(blue_unit.location)
            movements.append(direction)
            targets.append(target)

        assert all(len(x) <= self.action_length for x in [
                   locations, movements, targets])
        return (locations, movements, targets, recruitment)

    def act(self, observation: dict[str, any]):
        sample = self.__a2c.predict(self.observation_space)
        movements = sample[0:7]
        targets = sample[7:14]
        recruitment = sample[14]
        chunks = list(zip(movements, targets))
        self._set_ready_to_action(observation)
        return self._action(chunks, recruitment)


class RandomAgent(BaseAgent):
    def __init__(self, index: int, action_length: int = ACTION_LENGTH):
        super().__init__()
        self.action_length = action_length
        self._world = World(index, 1 - index)

    def action(self, observation: dict[str, any]):
        self._set_ready_to_action(observation)
        locations, movements, targets, = [], [], []
        blue_team = self._world.blue_team
        lower_bound = min(len(blue_team.units), self.action_length)
        for i in range(lower_bound):
            blue_unit = blue_team.units[i]
            available_directions = blue_unit.get_available_directions(
                self._world)
            if len(available_directions) == 0:
                continue

            movement_offsets = get_movement_offsets(blue_unit.location)
            target = calculate_location_by_offset(
                movement_offsets[random.randint(0, 6)], blue_unit.location)
            direction = random.choice(available_directions)
            if 0 < direction:
                coordinate = calculate_location_by_offset(
                    movement_offsets[direction], blue_unit.location)
                self._world.reserved_locations.append(coordinate)

            blue_unit.will_move = True
            locations.append(blue_unit.location)
            movements.append(direction)
            targets.append(target)

        assert all(len(x) <= self.action_length for x in [
                   locations, movements, targets])
        return (locations, movements, targets, random.randint(0, 4))
