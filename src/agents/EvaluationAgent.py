
import random
from enum import Enum

import numpy as np

from utilities import *


class UNIT_TAGS(Enum):
    HEAVY_TANK = "HeavyTank"
    LIGHT_TANK = "LightTank"
    TRUCK = "Truck"
    DRONE = "Drone"


class TERRAIN_TAGS(Enum):
    GRASS = 0
    DIRT = 1
    MOUNTAIN = 2
    WATER = 3


DIRECTION_OFFSETS_EVEN: dict[int, tuple[int, int]] = {
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
}
DIRECTION_OFFSETS_ODD: dict[int, tuple[int, int]] = {
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
    x, y = location[1], location[0]
    x_offset, y_offset = offset[0], offset[1]
    return (x + x_offset, y + y_offset)


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
            X, Y = calculate_location_by_offset(offset, self.location)
            coordinate = (Y, X)
            if not (0 <= Y < height and 0 <= X < width):
                continue

            if coordinate in [*world.reserved_locations,
                              world.blue_team.base_location,
                              world.red_team.base_location,
                              *[unit.location for unit in all_units]]:
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

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: UNIT_TAGS) -> None:
        super().__init__(hp, load, location, tag)

    def get_available_directions(self, world) -> list[int]:
        return super().get_available_directions(world)


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

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: UNIT_TAGS) -> None:
        super().__init__(hp, load, location, tag)

    def get_available_directions(self, world) -> list[int]:
        return super().get_available_directions(world)


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

    def get_available_directions(self, world) -> list[int]:
        return super().get_available_directions(world)


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

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: UNIT_TAGS) -> None:
        super().__init__(hp, load, location, tag)

    def get_available_directions(self, world) -> list[int]:
        return super().get_available_directions(world)


class Team:
    index: int
    units: list[Unit]
    base_location: tuple[int, int]

    def __init__(self, index: int) -> None:
        self.index = index
        self.units = []


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

    terrain: np.ndarray
    resources: np.ndarray

    reserved_locations: list[dict[int, int]] = []

    def clear(self):
        self.blue_team.units.clear()
        self.red_team.units.clear()
        self.reserved_locations.clear()


class EvaluationAgent:
    __action_length: int = -1
    __world: World

    def __init__(self, team, action_lenght=6):
        self.__action_lenght = action_lenght
        self.__world = World(team, 1 - team)

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

    def action(self, obs):
        self.__world.clear()

        self.__world.terrain = obs["terrain"]
        decoded_obs = decodeState(obs)
        for i, entity in enumerate(decoded_obs[self.__world.blue_team.index] + decoded_obs[self.__world.red_team.index]):
            unit = self.__get_unit(entity)
            if i < len(decoded_obs[self.__world.blue_team.index]):
                self.__world.blue_team.units.append(unit)
            else:
                self.__world.red_team.units.append(unit)

        self.__world.blue_team.base_location = decoded_obs[self.__world.blue_team.index + 2]
        self.__world.red_team.base_location = decoded_obs[self.__world.red_team.index + 2]
        self.__world.resources = decoded_obs[4]
        locations, movements, targets, = [], [], []
        train = 0
        for blue_unit in self.__world.blue_team.units:
            available_directions = blue_unit.get_available_directions(
                self.__world)
            if len(available_directions) == 0:
                continue

            direction = random.choice(available_directions)
            X, Y = calculate_location_by_offset(
                get_movement_offsets(blue_unit.location)[direction], blue_unit.location)
            self.__world.reserved_locations.append((Y, X))
            locations.append(blue_unit.location)
            movements.append(direction)
            targets.append(self.__world.red_team.units[0].location)

        assert all(len(x) <= self.__action_lenght for x in [
                   locations, movements, targets])
        return (locations, movements, targets, 0)
