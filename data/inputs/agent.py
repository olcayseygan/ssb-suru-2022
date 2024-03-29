from enum import Enum

import numpy as np
from gym.spaces import Box, MultiDiscrete

# from utilities import *


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

ACTION_LENGTH: int = 6


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


class EvaluationAgent():
    # Private:
    __world: World
    __state: dict

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

    # Public
    observation_space: Box
    action_space: MultiDiscrete

    def __init__(self, observation_space: Box, action_space: MultiDiscrete):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation):

        return self.action_space.sample()
