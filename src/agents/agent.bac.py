from argparse import Namespace
from enum import Enum

import numpy as np
from game import Game
from gym import Env, spaces

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

    def is_on_resource(self, resources: tuple[int, int]) -> bool:
        for resource in resources:
            if self.location == resource:
                return True

        return False

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


class EvaluationAgent(Env):
    __ACTION_LENGTH: int = 6
    __game: Game
    __world: World
    __episodes: int = 0
    __steps: int = 0
    __state: dict
    __reward: float = 0.0

    observation_space: spaces.Box
    action_space: spaces.Discrete

    def __init__(self, kwargs: Namespace, agents: list[str]) -> None:
        super().__init__()
        self.__game = Game(kwargs, agents)
        self.__world = World(0, 1)
        self.__world.blue_team.base.location = (
            self.__game.bases[self.__world.blue_team.index].y_coor,
            self.__game.bases[self.__world.blue_team.index].x_coor
        )
        self.__world.red_team.base.location = (
            self.__game.bases[self.__world.red_team.index].y_coor,
            self.__game.bases[self.__world.red_team.index].x_coor
        )
        self.__world.width = self.__game.map_x
        self.__world.height = self.__game.map_y
        height, width = self.__world.get_size()
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(height * width * 10 + 4,),
            dtype=np.int8
        )
        chunks = (
            7,
            7
        ) * self.__ACTION_LENGTH
        self.action_space = spaces.MultiDiscrete(chunks + (5,))

    def __flat_state(self, state):
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

    def __decode_state(self, state):
        units = state['units']
        hps = state['hps']
        bases = state['bases']
        res = state['resources']
        load = state['loads']

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

    def __get_unit(self, entity: list[any]) -> Unit:
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

    def action(self, chunks: list[tuple], train: int):
        self.__world.clear()

        self.__world.terrain = self.__state["terrain"]
        decoded_state = self.__decode_state(self.__state)
        scores = self.__state["score"]

        # Blue Team
        blue_team = self.__world.blue_team
        blue_team.base.load = int(scores[blue_team.index])
        blue_team.base.location = decoded_state[blue_team.index + 2]
        for i, entity in enumerate(decoded_state[blue_team.index]):
            unit = self.__get_unit(entity)
            blue_team.units.append(unit)

        # Red Team
        red_team = self.__world.red_team
        red_team.base.load = int(scores[red_team.index])
        red_team.base.location = decoded_state[red_team.index + 2]
        for i, entity in enumerate(decoded_state[red_team.index]):
            unit = self.__get_unit(entity)
            red_team.units.append(unit)

        self.__world.resources = decoded_state[4]
        locations, movements, targets, = [], [], []

        lower_bound = min(len(blue_team.units), len(chunks))
        for i in range(lower_bound):
            blue_unit = blue_team.units[i]
            chunk = chunks[i]
            available_directions = blue_unit.get_available_directions(
                self.__world)
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
                self.__world.reserved_locations.append(coordinate)

            blue_unit.will_move = True
            locations.append(blue_unit.location)
            movements.append(direction)
            targets.append(target)

        assert all(len(x) <= self.__ACTION_LENGTH for x in [
                   locations, movements, targets])
        return (locations, movements, targets, train)

    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec
        print("setup")

    def reset(self):
        self.__episodes += 1
        self.__steps = 0
        self.__state = self.__game.reset()
        self.__reward = 0.0
        return self.__flat_state(self.__state)

    def step(self, action):
        chunks = [action[:-1][i:i + 2] for i in range(0, len(action[:-1]), 2)]
        train = action[-1]
        self.__reward += 1
        action = self.action(chunks, train)
        self.__state, _, done = self.__game.step(action)
        self.__steps += 1

        reward = self.__reward
        reward += self.__world.blue_team.base.load ** 2
        reward += sum([unit.load for unit in self.__world.blue_team.units if isinstance(unit, TruckUnit)])
        return self.__flat_state(self.__state), reward, done, {}

    def render(self,):
        return None

    def close(self,):
        return None

    def decode_state(self, obs):
        state, info = self.__flat_state(obs)
        return state

    def take_action(self, action):
        return self.just_take_action(action, self.nec_obs, self.team)