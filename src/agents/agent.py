import random
from argparse import Namespace
from enum import Enum

import numpy as np
from game import Game
from gym import Env
from gym.spaces import Box, MultiDiscrete
from perlin_noise import PerlinNoise
from stable_baselines3 import A2C
from typing_extensions import Self

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


def get_distance_between_two_locations(source: tuple[int, int], target: tuple[int, int]) -> int:
    if source == None or target == None:
        return -1

    source = list(source)
    target = list(target)
    source[0] -= (source[1] + 1) // 2
    target[0] -= (target[1] + 1) // 2
    return (abs(source[0] - target[0]) + abs(source[1] - target[1]) + abs(source[0] + source[1] - target[0] - target[1])) // 2

class AStarNode:
    def __init__(self) -> None:
        self.goal: int = 0
        self.fring: int = 0
        self.heuristic: float = 0
        self.previous: "AStarNode" = None

    def reset_GFH(self) -> None:
        self.goal = 0
        self.fring = 0
        self.heuristic = 0
        self.previous = None

class Tile(AStarNode):
    class TAGS(Enum):
        GRASS = 0
        DIRT = 1
        MOUNTAIN = 2
        WATER = 3

    IS_MOVEABLE: bool = True
    IS_STICKY: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.tag = None
        self.neighbors: dict[int, Tile] = {}
        self.has_resource = False
        self.has_base = False
        self.base: "Base" = None
        self.location: tuple[int, int] = (-1, -1)
        self.unit: "Unit" = None

    def distance_to(self, other: "Tile") -> int:
        if other is None:
            return 100

        return get_distance_between_two_locations(self.location, other.location)

class GrassTile(Tile):
    def __init__(self) -> None:
        super().__init__()
        self.tag = self.TAGS.GRASS


class DirtTile(Tile):
    IS_STICKY: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.tag = self.TAGS.DIRT


class MountainTile(Tile):
    IS_MOVEABLE: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.tag = self.TAGS.MOUNTAIN


class WaterTile(Tile):
    IS_MOVEABLE: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.tag = self.TAGS.WATER


class Unit:
    class TAGS(Enum):
        TRUCK = "Truck"
        LIGHT_TANK = "LightTank"
        HEAVY_TANK = "HeavyTank"
        DRONE = "Drone"

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: TAGS) -> None:
        self._cost: int = -0
        self._attack: int = 0
        self._max_hp: int = 0
        self._heavy: bool = False
        self._fly: bool = False
        self._anti_air: bool = False
        self._location = location
        self._tile: Tile = None
        self.hp = hp
        self.load = load
        self.tag = tag
        self.idle = True
        self.going_tile: Tile = None
        self.marked_to_attack: dict[Unit.TAGS, "Unit"] = {}

    @property
    def attack(self) -> int:
        return self._attack

    @property
    def max_hp(self) -> int:
        return self._max_hp

    @property
    def location(self) -> tuple[int, int]:
        return self._location

    @property
    def tile(self) -> Tile:
        return self._tile

    @tile.setter
    def tile(self, tile: Tile):
        self._location = tile.location
        tile.unit = self
        self._tile = tile

    def get_available_neighbors(self, units: list[Self], reserved_tiles: list[Tile], tile: Tile = None, return_self: bool = False) -> dict[int, Tile]:
        if tile is None:
            tile = self.tile

        available_neighbors: dict[int, Tile] = {}
        for direction, neighbor in tile.neighbors.items():
            if neighbor in [*[reserved_tile for reserved_tile in reserved_tiles],
                        *[unit.tile for unit in units if unit.idle and unit.tile is not self._tile]]:
                continue

            if neighbor in [unit.going_tile for unit in units if not unit.idle]:
                continue


            available_neighbors[direction] = neighbor

        if return_self:
            available_neighbors[0] = tile

        return available_neighbors


    def get_nearest_unit(self, units: list[Self]) -> Self:
        return min(units, key=lambda unit: get_distance_between_two_locations(self.location, unit.location), default=None)

    def get_units_in_range(self, units: list[Self], range: int = 3) -> list[Self]:
        return [unit for unit in units if get_distance_between_two_locations(self.location, unit.location) <= range]

    def get_direction_from_location(self, location: tuple[int, int]) -> int:
        direction: int = 0
        for direct, tile in self._tile.neighbors.items():
            if tile.location == location:
                direction = direct
                break

        return direction

    def distance_to(self, other: "Unit") -> int:
        if other is None:
            return 100

        return self.tile.distance_to(other.tile)

    def flee(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        Y, X = 0, 0
        length = len(world.opponent_team.units)
        for unit in world.opponent_team.units:
            y, x = unit.location
            Y += y
            X += x

        if 0 < length:
            Y //= length
            X //= length

        path = world.find_path(self, world.terrain.tiles[Y][X])
        if len(path) == 0:
            return Unit.action(self, world)

        next_tile, *_ = path
        direction = self.get_direction_from_location(next_tile.location)
        return (self.location, direction, None)

    def action(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        return (self.location, 0, None)

    def move(self, world: "World", tile: Tile) -> tuple[tuple[int, int], int,  tuple[int, int]]:
        path = world.find_path(self, tile)
        if len(path) == 0:
            return Unit.action(self, world)

        next_tile, *_ = path
        direction = self.get_direction_from_location(next_tile.location)
        return (self.location, direction, None)


class AttackerUnit(Unit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)

    def attack(self, world: "World", unit: "Unit") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        if self.distance_to(unit) <= 2:
            unit.marked_to_attack[self.tag] = self
            return (self.location, 0, unit.location)

        return self.move(world, unit.tile)

    def attack_nearest(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        nearest_opponent_unit = world.get_nearest_unit(self, world.opponent_team.units)
        if nearest_opponent_unit is None:
            return super().action(world)

        return self.attack(world, nearest_opponent_unit)

    def attack_eaches(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        nearest_opponent_unit = world.get_nearest_unit(self, [unit for unit in world.opponent_team.units if self.tag not in unit.marked_to_attack])
        if nearest_opponent_unit is None:
            return super().action(world)

        return self.attack(world, nearest_opponent_unit)

    def attack_trucks(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        nearest_opponent_truck = world.get_nearest_unit(self, [unit for unit in world.opponent_team.units if unit.tag == Unit.TAGS.TRUCK])
        if nearest_opponent_truck is None:
            return super().action(world)

        return self.attack(world, nearest_opponent_truck)

    def attack_move_to_opponent_base(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        nearest_opponent_unit = world.get_nearest_unit(self, world.opponent_team.units)
        if nearest_opponent_unit is not None and self.distance_to(nearest_opponent_unit) <= 2:
            return self.attack(world, nearest_opponent_unit)

        return self.move(world, world.opponent_team.base.tile)

    def action(self, world: "World", action_type: int) -> tuple[tuple[int, int], int, tuple[int, int]]:
        action_space = [False] * 6
        action_space[action_type] = True
        [attack_nearest_action,
         attack_eaches_action,
         attack_trucks_action,
         attack_move_to_opponent_base_action,
         flee_action,
         kamikaze_action] = action_space

        if attack_nearest_action:
            return self.attack_nearest(world)

        if attack_eaches_action:
            return self.attack_eaches(world)

        if attack_trucks_action:
            return self.attack_trucks(world)

        if attack_move_to_opponent_base_action:
            return self.attack_move_to_opponent_base(world)

        if flee_action:
            return self.move(world, world.main_team.base.tile)

        if kamikaze_action:
            if world.get_nearest_tile(world.opponent_team.base.tile, [unit.tile for unit in world.main_team.units]) is self.tile:
                return self.attack_move_to_opponent_base(world)
            else:
                return self.move(world, world.opponent_team.base.tile)

        return super().action(world)

class HeavyTankUnit(AttackerUnit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self._cost: int = 2
        self._attack: int = 2
        self._max_hp: int = 4
        self._heavy: bool = True
        self._fly: bool = False
        self._anti_air: bool = False


class LightTankUnit(AttackerUnit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self._cost: int = 1
        self._attack: int = 2
        self._max_hp: int = 2
        self._heavy: bool = False
        self._fly: bool = False
        self._anti_air: bool = True


class TruckUnit(Unit):

    @property
    def max_load(self) -> int:
        return self._max_load

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self._cost: int = 1
        self._max_hp: int = 1
        self._max_load: int = 3
        self._heavy: bool = False
        self._fly: bool = False
        self._anti_air: bool = False

        self._is_resource_picked_up: bool = False
        self._is_resource_delivered: bool = False

    def is_resource_picked_up(self) -> bool:
        return self._is_resource_picked_up

    def is_resource_delivered(self) -> bool:
        return self._is_resource_delivered

    def has_space(self) -> bool:
        return self.load < self._max_load

    def has_load(self) -> bool:
        return 0 < self.load

    def is_empty(self) -> bool:
        return self.load == 0

    def is_full(self) -> bool:
        return self.load >= self._max_load

    def is_on_resource(self) -> bool:
        return self._tile.has_resource

    def is_on_base(self) -> bool:
        return self._tile.has_base

    def pick_up_resource(self) -> tuple[tuple[int, int], int,  tuple[int, int]]:
        self._is_resource_picked_up = True
        return (self.location, 0, self.location)

    def deliver_resource(self) -> tuple[tuple[int, int], int,  tuple[int, int]]:
        self._is_resource_delivered = True
        return (self.location, 0, self.location)

    def move_to_resource(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        resource_tile: Tile = world.get_nearest_resource_tile(self)
        if resource_tile is None:
            return super().action(world)

        return self.move(world, resource_tile)

    def return_to_base(self, world: "World") -> tuple[tuple[int, int], int,  tuple[int, int]]:
        base_tile: Tile = world.main_team.base.tile
        return self.move(world, base_tile)

    def action(self, world: "World", action_type: int) -> tuple[tuple[int, int], int,  tuple[int, int]]:
        action_space = [False for _ in range(3)]
        action_space[action_type] = True
        [do_nothing_action, deliver_action, call_bell_action] = action_space

        if self.is_empty() and len(world.resources) == 0:
            return self.move(world, world.opponent_team.base.tile)

        if self.is_on_resource() and self.has_space():
            return self.pick_up_resource()

        if self.is_on_base() and self.has_load():
            return self.deliver_resource()

        if deliver_action:
            if self.has_load():
                return self.return_to_base(world)

        if call_bell_action:
            return self.return_to_base(world)

        if self.has_space():
            if self.has_load() and self.tile.distance_to(world.main_team.base.tile) < self.tile.distance_to(world.get_nearest_resource_tile(self)):
                return self.return_to_base(world)

            return self.move_to_resource(world)

        if self.has_load() or len(world.resources) == 0:
            return self.return_to_base(world)

        return super().action(world)


class DroneUnit(AttackerUnit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self._cost: int = 1
        self._attack: int = 1
        self._max_hp: int = 1
        self._heavy: bool = False
        self._fly: bool = True
        self._anti_air: bool = True



class Base:
    def __init__(self) -> None:
        self._location: tuple[int, int] = None
        self._tile: Tile = None
        self.load: int = 0

    @property
    def location(self) -> tuple[int, int]:
        return self._location

    @property
    def tile(self) -> Tile:
        return self._tile

    @tile.setter
    def tile(self, tile: Tile) -> None:
        tile.base = self
        self._location = tile.location
        self._tile = tile


class Team:
    def __init__(self, index: int) -> None:
        self.index: int = index
        self.units: list[Unit] = []
        self.base: Base = None

    def find_unit(self, location: tuple[int, int]) -> Unit:
        return next((unit for unit in self.units if unit.location == location), None)

class Terrain:
    def __init__(self) -> None:
        self.__tiles: list[list[Tile]] = []

    @property
    def tiles(self) -> list[list[Tile]]:
        return self.__tiles


class World:
    def __init__(self, main_team_index: int, opponent_team_index: int) -> None:
        self.__main_team = Team(main_team_index)
        self.__opponent_team = Team(opponent_team_index)
        self.__terrain: Terrain = Terrain()
        self.__resources: list[Tile] = []
        self.first_run: bool = True
        self.height: int = -1
        self.width: int = -1
        self.length: int = -1
        self.reserved_tiles: list[Tile] = []

    @property
    def main_team(self) -> Team:
        return self.__main_team

    @property
    def opponent_team(self) -> Team:
        return self.__opponent_team

    @property
    def terrain(self) -> Terrain:
        return self.__terrain

    @terrain.setter
    def terrain(self, terrain: np.ndarray):
        self.__terrain.tiles.clear()
        self.height, self.width = terrain.shape
        tiles: list[list[Tile]] = self.__terrain.tiles
        for y in range(self.height):
            row: list[Tile] = []
            for x in range(self.width):
                index = terrain[y, x]
                tile_class = Tile
                if index == Tile.TAGS.GRASS.value:
                    tile_class = GrassTile
                elif index == Tile.TAGS.DIRT.value:
                    tile_class = DirtTile
                elif index == Tile.TAGS.MOUNTAIN.value:
                    tile_class = MountainTile
                elif index == Tile.TAGS.WATER.value:
                    tile_class = WaterTile

                tile = tile_class()
                tile.location = (y, x)
                row.append(tile)

            tiles.append(row)

        for y in range(self.height):
            for x in range(self.width):
                tile = tiles[y][x]
                coordinate: tuple[int, int] = (y, x)
                for direction, offset in get_movement_offsets(coordinate).items():
                    Y, X = calculate_location_by_offset(offset, coordinate)
                    if not (0 <= Y < self.height and 0 <= X < self.width):
                        continue

                    neighbor = tiles[Y][X]
                    if tile is neighbor:
                        continue

                    tile.neighbors[direction] = neighbor

        self.length = get_distance_between_two_locations(
            (0, 0), (self.height - 1, self.width - 1))

    @property
    def resources(self) -> list[Tile]:
        return self.__resources

    @resources.setter
    def resources(self, resources_locations: list[dict[int, int]]) -> None:
        self.__resources.clear()
        for location in resources_locations:
            Y, X = location
            tile = self.__terrain.tiles[Y][X]
            tile.has_resource = True
            self.__resources.append(tile)

    @property
    def size(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def height_width(self) -> int:
        return self.height * self.width

    def clear(self):
        self.main_team.units.clear()
        self.opponent_team.units.clear()
        self.reserved_tiles.clear()

    def __find_path(self, unit: Unit, goal_tile: Tile):
        open_set: list[Tile] = [self.__terrain.tiles[unit.location[0]][unit.location[1]]]
        closed_set: list[Tile] = []

        while len(open_set) != 0:
            current_tile = min(open_set, key=lambda node: node.fring, default=open_set[0])
            if current_tile is goal_tile:
                path: list[Tile] = []
                node: Tile = goal_tile
                while (node.previous is not None):
                    path.append(node)
                    node = node.previous

                [node_.reset_GFH() for node_ in set(closed_set + open_set)]
                return path[::-1]

            open_set.remove(current_tile)
            closed_set.append(current_tile)

            neighbors = {direction: neighbor for direction, neighbor in current_tile.neighbors.items() if neighbor not in closed_set}
            if goal_tile in neighbors.values():
                neighbors[goal_tile.direction] = goal_tile

            for neighbor in neighbors.values():
                g = current_tile.goal + get_distance_between_two_locations(current_tile.location, neighbor.location) * (10 if neighbor.tag == Tile.TAGS.MOUNTAIN else 1)
                is_pathable = False
                if neighbor in open_set:
                    if g < neighbor.goal:
                        neighbor.goal = g
                        is_pathable = True
                else:
                    neighbor.goal = g
                    is_pathable = True
                    open_set.append(neighbor)

                if is_pathable:
                    neighbor.heuristic = get_distance_between_two_locations(current_tile.location, goal_tile.location)
                    neighbor.fring = neighbor.goal + neighbor.heuristic
                    neighbor.previous = current_tile

        return []

    def find_path(self, unit: Unit, goal_tile: Tile) -> list[Tile]:
        open_set: list[Tile] = [unit.tile]
        closed_set: list[Tile] = []

        while len(open_set) != 0:
            current_tile = min(open_set, key=lambda node: node.fring, default=open_set[0])
            if current_tile is goal_tile:
                path: list[Tile] = []
                node: Tile = goal_tile
                while (node.previous is not None):
                    path.append(node)
                    node = node.previous

                [node_.reset_GFH() for node_ in set(closed_set + open_set)]
                return path[::-1]

            open_set.remove(current_tile)
            closed_set.append(current_tile)


            neighbors = {direction: neighbor for direction, neighbor in self.get_tiles_with_their_directions(unit, current_tile) if neighbor not in closed_set}
            for direction, neighbor in current_tile.neighbors.items():
                if neighbor is goal_tile:
                    neighbors[direction] = neighbor

            for neighbor in neighbors.values():
                g = current_tile.goal + get_distance_between_two_locations(current_tile.location, neighbor.location) * (10 if neighbor.tag == Tile.TAGS.MOUNTAIN else 1)
                is_pathable = False
                if neighbor in open_set:
                    if g < neighbor.goal:
                        neighbor.goal = g
                        is_pathable = True
                else:
                    neighbor.goal = g
                    is_pathable = True
                    open_set.append(neighbor)

                if is_pathable:
                    neighbor.heuristic = get_distance_between_two_locations(current_tile.location, goal_tile.location)
                    neighbor.fring = neighbor.goal + neighbor.heuristic
                    neighbor.previous = current_tile

        return []


    def get_tiles_with_their_directions(self, unit: Unit, tile: Tile = None) -> dict[int, Tile]:
        if tile is None:
            tile = unit.tile

        units: list[Unit] = self.main_team.units + self.opponent_team.units
        tiles: dict[int, Tile] = {}
        for direction, neighbor in tile.neighbors.items():
            if neighbor in [*self.reserved_tiles, *[unit_.tile for unit_ in units if unit_.idle and unit_.tile is not unit.tile]]:
                continue

            if neighbor in [unit.going_tile for unit in units if not unit.idle]:
                continue

            if unit.tag is Unit.TAGS.HEAVY_TANK and neighbor.IS_STICKY:
                continue

            if unit.tag is not Unit.TAGS.DRONE and not neighbor.IS_MOVEABLE:
                continue

            tiles[direction] = neighbor

        return tiles

    def get_tiles_unit_can_move_to_in_distance(self, unit: Unit, distance: int, tile: Tile = None):
        if tile is None:
            tile = unit.tile

        def get_tiles(tile_: Tile, tiles: list[Tile] = []):
            neigbors = self.get_tiles_with_their_directions(unit, tile=tile_).values()
            tiles_: list[Tile] = []
            for tile__ in neigbors:
                if tile__ in tiles:
                    continue

                distance_ = get_distance_between_two_locations(tile.location, tile__.location)
                if distance < distance_:
                    continue

                tiles_.append(tile__)

            if len(tiles_) == 0:
                return []

            tiles += tiles_
            for tile__ in tiles_:
                tiles_ += get_tiles(tile__, tiles)

            return tiles_

        return get_tiles(tile, [tile])

    def get_nearest_resource_tile(self, truck_unit: TruckUnit) -> Tile:
        resources = [resource for resource in self.resources if resource.unit is None]
        return min(resources, key=lambda tile: get_distance_between_two_locations(truck_unit.tile.location, tile.location), default=None)

    def get_nearest_unit(self, unit: Unit, units: list[Unit]) -> Unit:
        if not unit._anti_air:
            units = [unit_ for unit_ in units if unit_.tag is not Unit.TAGS.DRONE]

        if unit.tag is Unit.TAGS.DRONE:
            units = [unit_ for unit_ in units if unit_.tag is not Unit.TAGS.HEAVY_TANK]

        return min(units, key=lambda unit_: get_distance_between_two_locations(unit.tile.location, unit_.tile.location), default=None)

    def get_nearest_tile(self, tile: Tile, tiles: list[Tile]) -> Tile:
        return min(tiles, key=lambda target_tile: tile.distance_to(target_tile), default=None)

    def is_location_in_range(self, location: dict[int, int]) -> bool:
        Y, X = location
        return 0 <= Y < self.height and 0 <= X < self.width

    @staticmethod
    def generate_random_world_config(height, width) -> np.ndarray:
        def flip(arr):
            return arr[::-1, ::-1]

        terrain_characters = {
            0: 'w',
            1: 'd',
            2: 'g',
            3: 'm'
        }

        perlin = PerlinNoise(octaves=3)
        noise = np.array([[perlin([i/width, j/height])
                         for j in range(width)] for i in range(height // 2)])
        normalized_noise = (noise - np.min(noise)) / \
            (np.max(noise) - np.min(noise))

        game_map = np.ndarray(normalized_noise.shape, dtype=int)
        resource_map = np.zeros(game_map.shape, dtype=int)
        base_map = np.zeros(game_map.shape, dtype=int)

        game_map[0 <= normalized_noise] = 0
        game_map[0.2 < normalized_noise] = 1
        game_map[0.3 < normalized_noise] = 2
        game_map[0.7 < normalized_noise] = 1
        game_map[0.9 < normalized_noise] = 3

        grasses = [(y, x) for y, row in enumerate(game_map)
                   for x, col in enumerate(row) if col == 2]
        if len(grasses) == 0:
            return World.generate_random_world_config(height, width)

        base_location = grasses[random.randint(0, len(grasses) - 1)]
        base_map[base_location] = 1

        grasses.remove(base_location)
        for _ in range(random.randint(0, len(grasses) // 2)):
            y, x = random.choice(grasses)
            resource_map[y, x] = 1
            grasses.remove((y, x))

        unit_count = random.randint(0, min(len(grasses), 7))
        unit_map = np.zeros(game_map.shape, dtype=int)
        for _ in range(unit_count):
            (y, x) = random.choice(grasses)
            unit_map[y, x] = random.randint(1, 4)
            grasses.remove((y, x))


        unit_map = np.concatenate((-flip(unit_map), unit_map))
        base_map = np.concatenate((-flip(base_map), base_map))
        resource_map = np.concatenate((flip(resource_map), resource_map))
        game_map = np.concatenate((flip(game_map), game_map))

        blue_base_location_y, blue_base_location_x = np.array(
            np.where(base_map == 1)).flatten()
        red_base_location_y, red_base_location_x = np.array(
            np.where(base_map == -1)).flatten()

        blue_units_locations = np.where(0 < unit_map)
        red_units_locations = np.where(unit_map < 0)

        if game_map[blue_base_location_y, blue_base_location_x] != 2:
            return World.generate_random_world_config(height, width)

        return {
            "max_turn": random.randint(32, 64),
            "turn_timer": 10,
            "map": {
                "x": width,
                "y": height,
                "terrain": np.vectorize(terrain_characters.get)(game_map).tolist()
            },
            "red": {
                "base": {
                    "x": red_base_location_x,
                    "y": red_base_location_y,
                },
                "units": [
                    {
                        "type": ["Truck", "LightTank", "HeavyTank","Drone"][abs(unit_map[y][x]) - 1],
                        "x": x,
                        "y": y,
                    }
                    for y, x in zip(*red_units_locations)
                ]
            },
            "blue": {
                "base": {
                    "x": blue_base_location_x,
                    "y": blue_base_location_y,
                },
                "units": [
                    {
                        "type": ["Truck", "LightTank", "HeavyTank","Drone"][unit_map[y][x] - 1],
                        "x": x,
                        "y": y,
                    }
                    for y, x in zip(*blue_units_locations)
                ]
            },
            "resources": [
                {
                    "x": x,
                    "y": y
                } for y, x in np.argwhere(resource_map == 1).tolist()
            ]
        }


class Agent():
    def _create_unit(self, entity: list[any]) -> Unit:
        [_, tag, hp, location, load] = entity.values()
        if tag == Unit.TAGS.HEAVY_TANK.value:
            unit = HeavyTankUnit(hp, load, location, Unit.TAGS.HEAVY_TANK)
        elif tag == Unit.TAGS.LIGHT_TANK.value:
            unit = LightTankUnit(hp, load, location, Unit.TAGS.LIGHT_TANK)
        elif tag == Unit.TAGS.TRUCK.value:
            unit = TruckUnit(hp, load, location, Unit.TAGS.TRUCK)
        elif tag == Unit.TAGS.DRONE.value:
            unit = DroneUnit(hp, load, location, Unit.TAGS.DRONE)
        else:
            unit = Unit(hp, load, location, Unit.TAGS(tag))

        return unit

    def _find_unit(self, location: dict[int, int]) -> Unit:
        world = self._world
        units: list[Unit] = world.main_team.units + world.opponent_team.units
        for unit in units:
            if unit.going_tile is not None:
                if unit.going_tile.location == location:
                    return unit
            else:
                if unit.location == location:
                    return unit

        return None

    def _decode_state(self, observation: dict[str, any]):
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

    def _action(self, estimations: list[int], recruitment: int):
        world = self._world
        actions = np.ndarray(shape=(0, 3), dtype=object)

        # needed variables
        main_team = world.main_team
        opponent_team = world.opponent_team
        main_team_units = main_team.units
        # opponent_team_units = opponent_team.units
        # sorted_main_team_units = sorted(main_team_units, key=lambda unit: get_distance_between_two_locations(
        #     main_team.base.location, unit.location))

        truck_estimations = estimations[:1]
        drone_estimations = estimations[1:2]
        heavy_tank_estimations = estimations[2:3]
        light_tank_estimations = estimations[3:4]
        for unit in main_team_units:
            if unit.tag is Unit.TAGS.TRUCK:
                action_type = truck_estimations[0]
            elif unit.tag is Unit.TAGS.DRONE:
                action_type = drone_estimations[0]
            elif unit.tag is Unit.TAGS.HEAVY_TANK:
                action_type = heavy_tank_estimations[0]
            elif unit.tag is Unit.TAGS.LIGHT_TANK:
                action_type = light_tank_estimations[0]

            source_location, direction, target_location = unit.action(world, action_type)
            """
            # unit.action(world)
            # target_tile = None
            # if unit.tag == Unit.TAGS.TRUCK:
            #     nearest_resource = world.get_nearest_resource_tile(unit)
            #     if nearest_resource is not None:
            #         target_tile = nearest_resource
            # else:
            #     # target_tile = world.opponent_team.base.on_tile
            #     nearest_unit = world.get_nearest_unit(unit, opponent_team_units)
            #     if nearest_unit is not None:
            #         target_tile = nearest_unit.tile

            # if target_tile is None:
            #     continue

            # direction_tile = unit.tile
            # path = world.find_path(unit, target_tile)
            # if len(path) == 0:
            #     continue

            # direction_tile, *_ = path
            # if direction_tile is None:
            #     continue

            # direction = unit.get_direction_from_location(direction_tile.location)
            # target = None

            # if unit is not Unit.TAGS.TRUCK:
            #     nearest_oppenent_unit = world.get_nearest_unit(unit, opponent_team_units)
            #     if unit.distance_to(nearest_oppenent_unit) <= 2:
            #         direction = 0
            #         target = nearest_oppenent_unit.location

            # checking that there are some directions to action
            # available_neighbors = unit.get_available_neighbors(
            #     [*main_team_units, *opponent_team_units],
            #     world.reserved_tiles
            # )
            # if len(available_neighbors.keys()) == 0:
            #     continue

            # offset = estimation["offset"]
            # target = (unit.location[0] + offset[0], unit.location[1] + offset[1])
            # if not world.is_location_in_range(target):
            #     continue

            # target = None
            # direction = unit.get_direction_from_location(direction_tile.location)
            # sorted_opponent_units_to_unit = sorted(opponent_team.units, key=lambda unit: get_distance_between_two_locations(
            #     unit.location, unit.location))
            # if unit.tag is Unit.TAGS.TRUCK:
            #     target = unit.location if direction == 0 else None
            #     if target is not None:
            #         if unit.has_load() and unit.is_on_base():
            #             unit.deliver_resource()
            #         elif unit.has_space() and unit.is_on_resource():
            #             unit.pick_up_resource()
            #         else:
            #             target = None

            # if target is not None:
            #     if unit.tag is not Unit.TAGS.TRUCK:
            #         distance_to_target = get_distance_between_two_locations(
            #             unit.location, target)
            #         if 2 < distance_to_target:
            #             target = None

            # if 0 < direction:
            #     world.reserved_tiles.append(unit.tile.neighbors[direction])
            # else:
            #     if unit.tag == Unit.TAGS.TRUCK:
            #         if (unit.has_space() and unit.is_on_resource()) or (unit.has_load() and unit.is_on_base()):
            #             target = unit.location

            # if target is None and direction == 0:
            #     continue
            """

            unit.idle = False
            unit.going_tile = unit.tile.neighbors[direction] if 0 < direction else unit.tile
            actions = np.append(actions, [[source_location, direction, target_location]], axis=0)

        if self.action_length <= len(world.main_team.units):
            recruitment = 0

        assert len(actions) <= self.action_length
        source_locations, directions, target_directions, = actions[:, 0], actions[:, 1], actions[:, 2]
        return (source_locations, directions, target_directions, recruitment)

    def _flat_state_2(self, state: dict[str, any]):
        score: np.ndarray = np.array(state["score"], dtype=np.int8)
        turn: int = state["turn"]
        max_turn: int = state["max_turn"]
        units: np.ndarray = np.array(state["units"], dtype=np.int8)
        hps: np.ndarray = np.array(state["hps"], dtype=np.int8)
        bases: np.ndarray = np.array(state["bases"], dtype=np.int8)
        resources: np.ndarray = np.array(state["resources"], dtype=np.int8)
        loads: np.ndarray = np.array(state["loads"], dtype=np.int8)
        terrain: np.ndarray = np.array(state["terrain"], dtype=np.int8)

        return [
            *score,
            turn,
            max_turn,
            *(units[0] - units[1]).flatten(),
            *(hps[0] - hps[1]).flatten(),
            *(bases[0] - bases[1]).flatten(),
            *resources.flatten(),
            *(loads[0] - loads[1]).flatten(),
            *terrain.flatten()
        ]

    def __init__(self):
        self._is_map_built: bool = False
        self._world: World
        self._state: dict[str, any]
        self.action_length: int = ACTION_LENGTH
        pass

    def _build_map(self, state: dict[str, any]):
        if self._is_map_built:
            return

        world = self._world
        decoded_state = self._decode_state(state)
        world.terrain = state["terrain"]
        world.resources = decoded_state[4]
        main_team = self._world.main_team
        main_team.base = Base()
        base_y, base_x = decoded_state[main_team.index + 2]
        main_team.base.tile = world.terrain.tiles[base_y][base_x]
        opponent_team = self._world.opponent_team
        opponent_team.base = Base()
        base_y, base_x = decoded_state[opponent_team.index + 2]
        opponent_team.base.tile = world.terrain.tiles[base_y][base_x]
        self._is_map_built = True

    def _set_ready_to_action(self, state: dict[str, any]) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]], int]:
        world = self._world
        decoded_state = self._decode_state(state)
        scores = state["score"]

        world.clear()
        # Main Team
        main_team = self._world.main_team
        for i, entity in enumerate(decoded_state[main_team.index]):
            [_, _, _, location, _] = entity.values()
            y, x = location
            tile = world.terrain.tiles[y][x]
            unit = self._create_unit(entity)
            unit.tile = tile
            tile.unit = unit
            main_team.units.append(unit)

        # Oppenent Team
        opponent_team = self._world.opponent_team
        for i, entity in enumerate(decoded_state[opponent_team.index]):
            [_, _, _, location, _] = entity.values()
            y, x = location
            tile = world.terrain.tiles[y][x]
            unit = self._create_unit(entity)
            unit.tile = tile
            tile.unit = unit
            opponent_team.units.append(unit)

class TrainAgentEnv(BaseAgent, Env):
    def __world_to_dict(self, state: dict[str, any]) -> dict[str, any]:
        self._set_ready_to_action(state)
        return {
            "resources" : self._world.resources,
            "main": {
                "team": {
                    "base": {
                        "load": self._world.main_team.base.load,
                        "trained_unit": False
                    },
                    "units": {
                        "all": [unit for unit in self._world.main_team.units],
                        "trucks": [unit for unit in self._world.main_team.units if unit.tag == Unit.TAGS.TRUCK],
                        "heavy_tanks": [unit for unit in self._world.main_team.units if unit.tag == Unit.TAGS.HEAVY_TANK],
                        "light_tanks": [unit for unit in self._world.main_team.units if unit.tag == Unit.TAGS.LIGHT_TANK],
                        "drones": [unit for unit in self._world.main_team.units if unit.tag == Unit.TAGS.DRONE],
                    },
                    "killed_unit_count": 0,
                    "died_unit_count": 0,
                }
            },
            "opponent": {
                "team": {
                    "base": {
                        "load": self._world.opponent_team.base.load,
                        "trained_unit": False
                    },
                    "units": {
                        "all": [unit for unit in self._world.opponent_team.units],
                        "trucks": [unit for unit in self._world.opponent_team.units if unit.tag == Unit.TAGS.TRUCK],
                        "heavy_tanks": [unit for unit in self._world.opponent_team.units if unit.tag == Unit.TAGS.HEAVY_TANK],
                        "light_tanks": [unit for unit in self._world.opponent_team.units if unit.tag == Unit.TAGS.LIGHT_TANK],
                        "drones": [unit for unit in self._world.opponent_team.units if unit.tag == Unit.TAGS.DRONE],
                    },
                    "killed_unit_count": 0,
                    "died_unit_count": 0,
                }
            }
        }

    def __generate_world_config(self) -> dict[str, any]:
        height, width =  self.__game.map_y, self.__game.map_x
        self.__game.config = World.generate_random_world_config(height, width)
        # self.__game.max_turn = self.__game.config["max_turn"]
        # self.__game.turn_timer = self.__game.config["turn_timer"]
        return self.__game.reset()

    def __init__(self, kwargs: Namespace, agents: list[str]) -> None:
        super().__init__()
        self.__previous_state: dict[str, any] = {}
        self.__reward: float = 0.0
        self.__game = Game(kwargs, agents)
        self.__state: dict[str, any] = {}
        height, width =  self.__game.map_y, self.__game.map_x
        self._world = World(0, 1)
        length = height * width
        self.observation_space = Box(
            low=-32,
            high=512,
            shape=(length * 6 + 4,),
            dtype=np.int16
        )

        # TODO: all drone units should try to group up
        # TODO: scatter if there are so many dronein the map
        # TODO: optimize searching algorithm for tank and drone
        # TODO: denfend base
        # TODO: blocking the opponent unit if it is alone in the map if resources are exist
        self.action_space = MultiDiscrete(
            [3] # [do-nothing, deliver, call-bell] Truck Actions
            + [6] # [do-nothing, attack-nearby, attack-eaches, attack-trucks, attack-move-base, flee, kamikaze] Drone Actions8
            + [6] # [do-nothing, attack-nearby, attack-eaches, attack-trucks, attack-move-base, flee, kamikaze] Heavy Tank Actions
            + [6] # [do-nothing, attack-nearby, attack-eaches, attack-trucks, attack-move-base, flee, kamikaze] Light Tank Actions
            + [5])

    def setup(self, observation_space: Box, action_space: MultiDiscrete):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        self.__state = self.__generate_world_config()
        self.__reward = 0.0
        return self._flat_state_2(self.__state)

    def step(self, actions: np.ndarray):
        self._build_map(self.__state)
        self.__previous_state = self.__state
        previous_world: dict[str, any] = self.__world_to_dict(self.__previous_state)
        estimations = actions[:-1]
        recruitment = actions[-1]
        self.__state, _, done = self.__game.step(self._action(estimations, recruitment))
        current_world: dict[str, any] = self.__world_to_dict(self.__state)

        reward = 0.0
        # make a reward function
        # for unit in previous_world["main"]["team"]["units"]["all"]:
        #     if next((True for estimation in estimations if estimation == unit.location), False):
        #         reward += 0.2

        reward += current_world["main"]["team"]["base"]["load"]
        reward -= current_world["opponent"]["team"]["base"]["load"] / 2

        diff_between_unit_counts = len(previous_world["main"]["team"]["units"]["all"]) - len(current_world["main"]["team"]["units"]["all"])
        current_world["main"]["team"]["died_unit_count"] = abs(diff_between_unit_counts) if diff_between_unit_counts < 0 else 0
        diff_between_unit_counts = len(previous_world["opponent"]["team"]["units"]["all"]) - len(current_world["opponent"]["team"]["units"]["all"])
        current_world["opponent"]["team"]["died_unit_count"] = abs(diff_between_unit_counts) if diff_between_unit_counts < 0 else 0

        reward += current_world["opponent"]["team"]["died_unit_count"] / 14
        reward -= current_world["main"]["team"]["died_unit_count"] / 7


        """# Gold Collection
        for unit in previous_world["main"]["team"]["units"]["trucks"]:
            # reward += unit.load * 8
            if unit.is_resource_picked_up():
                reward += 2
            elif unit.is_resource_delivered():
                reward += 8

            if unit.has_space():
                nearest_resource = unit.get_nearest_resource(previous_world["resources"])
                if nearest_resource is not None:
                    distance_to_resource = get_distance_between_two_locations(unit.location, nearest_resource.location)
                    reward += (1 - distance_to_resource / self._world.length) * 0.1 * (unit.load + 1)

            if unit.has_load():
                distance_to_base = get_distance_between_two_locations(unit.location, self._world.main_team.base.location)
                reward += (1 - distance_to_base / self._world.length) * (0.2 if unit.is_full() else 0.8) * unit.load

        # Main            --------------------------------------------------------------------------------------------------------------------------------
        reward += previous_world["main"]["team"]["base"]["load"] * 32
        diff_between_unit_counts = len(previous_world["main"]["team"]["units"]["all"]) - len(current_world["main"]["team"]["units"]["all"])
        current_world["main"]["team"]["died_unit_count"] = abs(diff_between_unit_counts) if diff_between_unit_counts < 0 else 0
        current_world["main"]["team"]["base"]["trained_unit"] = 0 < diff_between_unit_counts
        # if current_world["main"]["team"]["base"]["trained_unit"]:
        #     reward += 4

        reward -= 4 * current_world["main"]["team"]["died_unit_count"]
        for unit in current_world["main"]["team"]["units"]["all"]:
            reward -= unit.max_hp - unit.hp

        for unit in previous_world["main"]["team"]["units"]["trucks"]:
            opponent_units_in_range = [unit_ for unit_ in unit.get_units_in_range(previous_world["main"]["team"]["units"]["all"], 3) if unit_.tag is not Unit.TAGS.TRUCK]
            nearest_enemy = unit.get_nearest_unit(opponent_units_in_range)
            if nearest_enemy is None:
                continue

            for unit_in_range in opponent_units_in_range:
                reward -= 0.08 * (unit_in_range.hp ** unit_in_range.attack) * (1 - get_distance_between_two_locations(unit.location, unit_in_range.location) / 3)

            main_units_in_range = [unit_ for unit_ in unit.get_units_in_range(previous_world["main"]["team"]["units"]["all"], 5) if unit_.tag is not Unit.TAGS.TRUCK]
            for unit_in_range in main_units_in_range:
                reward += 0.8 * (unit_in_range.hp ** unit_in_range.attack) * (1 - get_distance_between_two_locations(unit_in_range.location, nearest_enemy.location) / self._world.length)

        # Opponent        --------------------------------------------------------------------------------------------------------------------------------
        reward -= previous_world["opponent"]["team"]["base"]["load"] * 40
        diff_between_unit_counts = len(previous_world["opponent"]["team"]["units"]["all"]) - len(current_world["opponent"]["team"]["units"]["all"])
        current_world["opponent"]["team"]["died_unit_count"] = abs(diff_between_unit_counts) if diff_between_unit_counts < 0 else 0
        current_world["opponent"]["team"]["base"]["trained_unit"] = 0 < diff_between_unit_counts
        if current_world["opponent"]["team"]["base"]["trained_unit"]:
            reward -= 4

        reward += 8 * current_world["opponent"]["team"]["died_unit_count"]
        for unit in current_world["opponent"]["team"]["units"]["all"]:
            reward += (unit.max_hp - unit.hp) ** 2"""

        self.__reward += reward
        # print(f"{self.__reward:0.4f}, {reward:0.4f}")
        return self._flat_state_2(self.__state), self.__reward, done, {}

    def render(self,):
        return None

    def close(self,):
        return None


class EvaluationAgent(Agent):
    """An example agent which shares the same learning infrastructure as
    the full-fledged benchmarks, but implements random action selection."""

    # Private:
    __a2c: A2C

    # Public:
    observation_space: Box
    action_space: MultiDiscrete

    def __init__(self, observation_space_or_team_index: any, action_space_or_action_length: any):
        super().__init__()
        if isinstance(observation_space_or_team_index, int):
            main_team_index = observation_space_or_team_index
            opponent_team_index = 1 - observation_space_or_team_index
            self._world = World(main_team_index, opponent_team_index)
        else:
            self._world = World(0, 1)

        self.__a2c = A2C.load(
            ".\\models\\v1\\TrainSingleMixedLarge_4000_steps")
        self.observation_space = self.__a2c.observation_space
        self.action_space = self.__a2c.action_space

    def action(self, observation: dict[str, any]):
        return self.act(observation)

    def act(self, state: dict[str, any]):
        self._set_ready_to_action(state)
        actions, *_ = self.__a2c.predict(self._flat_state_2(state))
        estimations = actions[:-1]
        recruitment = actions[-1]
        return self._action(estimations, recruitment)

class RandomAgent(Agent):
    def __init__(self, index: int, action_length: int = ACTION_LENGTH):
        super().__init__()
        self.action_length = ACTION_LENGTH
        self._world = World(index, 1 - index)

    def action(self, observation: dict[str, any]):
        self._set_ready_to_action(observation)
        world = self._world
        locations, movements, targets, units, = [], [], [], []
        blue_team = world.main_team
        red_team = world.opponent_team
        lower_bound = min(len(blue_team.units), self.action_length)
        for i in range(lower_bound):
            blue_unit = blue_team.units[i]
            available_neighbors = blue_unit.get_available_neighbors(
                [*blue_team.units, *red_team.units],
                world.reserved_tiles
            )
            if len(available_neighbors.keys()) == 0:
                continue

            direction = random.choice(list(available_neighbors.keys()))
            target = blue_unit.location
            (Y, X) = target
            if 0 < direction:
                world.reserved_tiles.append(world.terrain.tiles[Y][X])

            blue_unit.going_tile = blue_unit.tile.neighbors[direction]
            blue_unit.idle = False
            locations.append(blue_unit.location)
            units.append(blue_unit)
            movements.append(direction)
            targets.append(target)

        # assert all(len(x) <= self.action_length for x in [
        #            locations, movements, targets])
        return (locations, movements, targets, random.randint(0, 4))
