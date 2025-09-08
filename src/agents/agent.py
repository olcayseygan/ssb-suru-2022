import random
from argparse import Namespace
from enum import Enum
from typing import Dict, Tuple, Optional, Any, List
import numpy.typing as npt

import numpy as np
from game import Game
from gym import Env
from gym.spaces import Box, MultiDiscrete
from perlin_noise import PerlinNoise
from stable_baselines3 import A2C

from utilities import *

DIRECTION_OFFSETS_EVEN: Dict[int, Tuple[int, int]] = {
    0: (0, 0),
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
}
DIRECTION_OFFSETS_ODD: Dict[int, Tuple[int, int]] = {
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


def get_movement_offsets(location: Tuple[int, int]) -> Dict[int, Tuple[int, int]]:
    return DIRECTION_OFFSETS[location[1] % 2]


class AStarNode:
    def __init__(self) -> None:
        self.goal: int = 0
        self.fring: int = 0
        self.heuristic: float = 0
        self.previous: Optional["AStarNode"] = None

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
        self.base: Optional["Base"] = None
        self.has_base = False
        self.has_resource = False
        self.location: Tuple[int, int] = (-1, -1)
        self.neighbors: Dict[int, Tile] = {}
        self.tag: Optional["Tile.TAGS"] = None
        self.unit: Optional["Unit"] = None
        self.world: Optional["World"] = None

    def distance_to(self, target_tile: "Tile") -> int:
        if target_tile is None:
            return 100

        source = list(self.location)
        target = list(target_tile.location)
        source[0] -= (source[1] + 1) // 2
        target[0] -= (target[1] + 1) // 2
        return (abs(source[0] - target[0]) + abs(source[1] - target[1]) + abs(source[0] + source[1] - target[0] - target[1])) // 2

    def get_tile_by_offset(self, offset: Tuple[int, int]) -> Optional["Tile"]:
        y, x = self.location
        x_offset, y_offset = offset
        Y, X = y + y_offset, x + x_offset
        if not self.world:
            return None

        if not (0 <= Y < self.world.height and 0 <= X < self.world.width):
            return None

        if not self.world.terrain:
            return None

        return self.world.terrain.tiles[Y][X]

    def get_nearest_tile(self, tiles: list["Tile"]) -> "Tile":
        return min(tiles, key=lambda tile: self.distance_to(tile), default=None)

    def get_nearest_resource_tile(self) -> "Tile":
        return self.get_nearest_tile(self.world.terrain.resource_tiles)

    def get_tiles_able_to_move(self, unit: Optional["Unit"] = None) -> Dict[int, "Tile"]:
        if unit is None:
            unit = self.unit

        tiles: Dict[int, "Tile"] = {}
        if not self.world:
            return tiles

        units: list["Unit"] = self.world.units
        for direction, neighbor in self.neighbors.items():
            if neighbor in [*self.world.reserved_tiles, *[unit_.tile for unit_ in units if unit_.idle and unit and unit_.tile is not unit.tile]]:
                continue

            if neighbor in [unit.going_tile for unit in units if not unit.idle]:
                continue

            if not unit:
                continue

            if unit.tag is Unit.TAGS.HEAVY_TANK and neighbor.IS_STICKY:
                continue

            if unit.tag is not Unit.TAGS.DRONE and not neighbor.IS_MOVEABLE:
                continue

            tiles[direction] = neighbor

        return tiles

    def get_tiles_able_to_move_in_distance(self, distance: int, can_have_resource: bool = True) -> list["Tile"]:
        def get_tiles(tile_: Tile, tiles: list[Tile] = []):
            neigbors = tile_.get_tiles_able_to_move(self.unit).values()
            tiles_: list[Tile] = []
            for tile__ in neigbors:
                if not can_have_resource:
                    if tile__.has_resource:
                        continue

                if tile__ in tiles:
                    continue

                distance_ = self.distance_to(tile__)
                if distance < distance_:
                    continue

                tiles_.append(tile__)

            if len(tiles_) == 0:
                return []

            tiles += tiles_
            for tile__ in tiles_:
                tiles_ += get_tiles(tile__, tiles)

            return tiles_

        return get_tiles(self, [self])

    def get_direction_by_tile(self, tile: "Tile") -> int:
        for direction, neighbor in self.neighbors.items():
            if neighbor is tile:
                return direction

        return 0

    def get_tile_by_direction(self, direction: int) -> Optional["Tile"]:
        return self.neighbors.get(direction, None)


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

    def __init__(self, hp: int, load: int, location: Tuple[int, int], tag: TAGS) -> None:
        self._tile: Optional["Tile"] = None
        self.anti_air: bool = False
        self.cost: int = 0
        self.fly: bool = False
        self.going_tile: Optional["Tile"] = None
        self.heavy: bool = False
        self.hp = hp
        self.idle = True
        self.load = load
        self.marked_to_attack: dict[Unit.TAGS, "Unit"] = {}
        self.max_hp: int = 0
        self.tag = tag

    @property
    def location(self) -> Optional[Tuple[int, int]]:
        if not self.tile:
            return None

        return self.tile.location

    @property
    def tile(self) -> Optional["Tile"]:
        return self._tile

    @tile.setter
    def tile(self, tile: "Tile"):
        tile.unit = self
        self._tile = tile

    def do_nothing(self) -> Optional[Tuple[Tuple[int, int], int, None]]:
        if not self.location:
            return None

        return (self.location, 0, None)

    def move(self, direction: int) -> Optional[Tuple[Tuple[int, int], int, None]]:
        if not self.location:
            return self.do_nothing()

        return (self.location, direction, None)

    def action(self, movement: int, targeting: int) -> Optional[Tuple[Tuple[int, int], int, Optional[Tuple[int, int]]]]:
        if not self.tile:
            return self.do_nothing()

        target_tile = self.tile.get_tile_by_direction(targeting)
        if not target_tile:
            return self.do_nothing()

        if not self.location:
            return self.do_nothing()

        available_tiles = self.tile.get_tiles_able_to_move(self)
        if movement not in available_tiles:
            return self.do_nothing()

        return (self.location, movement, target_tile.location)


class AttackerUnit(Unit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)

    def action(self, movement: int, targeting: int) -> Optional[Tuple[Tuple[int, int], int, Optional[Tuple[int, int]]]]:
        return super().action(movement, targeting)


class HeavyTankUnit(AttackerUnit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self.anti_air: bool = False
        self.attack: int = 2
        self.cost: int = 2
        self.fly: bool = False
        self.heavy: bool = True
        self.max_hp: int = 4


class LightTankUnit(AttackerUnit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self.anti_air: bool = True
        self.attack: int = 2
        self.cost: int = 1
        self.fly: bool = False
        self.heavy: bool = False
        self.max_hp: int = 2


class TruckUnit(Unit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self.anti_air: bool = False
        self.cost: int = 1
        self.fly: bool = False
        self.heavy: bool = False
        self.max_hp: int = 1
        self.max_load: int = 3

    def pick_up_resource(self) -> Optional[Tuple[Tuple[int, int], int, Optional[Tuple[int, int]]]]:
        if not self.location:
            return self.do_nothing()

        return (self.location, 0, self.location)

    def deliver_resource(self) -> Optional[Tuple[Tuple[int, int], int, Optional[Tuple[int, int]]]]:
        if not self.location:
            return self.do_nothing()

        return (self.location, 0, self.location)

    def action(self, movement: int, targeting: int) -> Optional[Tuple[Tuple[int, int], int, Optional[Tuple[int, int]]]]:
        return super().action(movement, targeting)


class DroneUnit(AttackerUnit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self.cost: int = 1
        self.attack: int = 1
        self.max_hp: int = 1
        self.heavy: bool = False
        self.fly: bool = True
        self.anti_air: bool = True


class Base:
    def __init__(self) -> None:
        self._tile: Optional["Tile"] = None
        self.load: int = 0

    @property
    def location(self) -> Optional[Tuple[int, int]]:
        if not self.tile:
            return None

        return self.tile.location

    @property
    def tile(self) -> Optional["Tile"]:
        return self._tile

    @tile.setter
    def tile(self, tile: "Tile") -> None:
        tile.base = self
        self._tile = tile


class Team:
    def __init__(self, index: int) -> None:
        self.index: int = index
        self.units: List["Unit"] = []
        self.base: Optional["Base"] = None


class Terrain:
    def __init__(self) -> None:
        self.tiles: List[List["Tile"]] = []
        self.resource_tiles: List["Tile"] = []

    def flatten(self) -> List["Tile"]:
        return [tile for row in self.tiles for tile in row]

    def set_resource_tiles(self, resource_locations: List[Tuple[int, int]]) -> None:
        for tile in self.flatten():
            tile.has_resource = False

        self.resource_tiles.clear()
        for resource_location in resource_locations:
            Y, X = resource_location
            tile = self.tiles[Y][X]
            tile.has_resource = True
            self.resource_tiles.append(tile)


class World:
    def __init__(self, main_team_index: int, opponent_team_index: int) -> None:
        self.resources: List["Tile"] = []
        self.terrain: Terrain = Terrain()
        self.first_run: bool = True
        self.height: int = -1
        self.length: int = -1
        self.main_team = Team(main_team_index)
        self.opponent_team = Team(opponent_team_index)
        self.reserved_tiles: List["Tile"] = []
        self.width: int = -1

    def set_terrain(self, terrain: np.ndarray, resources_locations: List[Tuple[int, int]]) -> None:
        self.terrain.tiles.clear()
        self.height, self.width = terrain.shape
        tiles: List[List["Tile"]] = self.terrain.tiles
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
                tile.world = self
                row.append(tile)

            tiles.append(row)

        for y in range(self.height):
            for x in range(self.width):
                tile = tiles[y][x]
                coordinate: tuple[int, int] = (y, x)
                for direction, offset in get_movement_offsets(coordinate).items():
                    neighbor = tile.get_tile_by_offset(offset)
                    if neighbor is None:
                        continue

                    if tile is neighbor:
                        continue

                    tile.neighbors[direction] = neighbor

        self.length = tiles[0][0].distance_to(tiles[-1][-1])
        self.terrain.set_resource_tiles(resources_locations)

    @property
    def units(self) -> list[Unit]:
        return self.main_team.units + self.opponent_team.units

    def clear(self):
        self.main_team.units.clear()
        self.opponent_team.units.clear()
        self.reserved_tiles.clear()
        for tile in self.terrain.flatten():
            tile.unit = None

    @staticmethod
    def generate_random_world_config(height: int, width: int) -> Dict[str, Any]:
        def flip(arr):
            return arr[::-1, ::-1]

        terrain_characters = {
            0: 'w',
            1: 'd',
            2: 'g',
            3: 'm'
        }

        perlin = PerlinNoise(octaves=3)
        noise = np.array([[perlin([i/width, j/height]) for j in range(width)] for i in range(height // 2)])
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
    def __init__(self, kwargs: Namespace, agents: list[str]) -> None:
        self._kwargs = kwargs
        self._is_map_built: bool = False
        self._state: Dict[str, Any] = {}
        self._world: Optional["World"] = None
        self._game = Game(kwargs, agents)
        self.action_length: int = ACTION_LENGTH

    def _world_to_dict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        world = self._world
        if not world:
            return {}

        if not world.main_team.base or not world.opponent_team.base:
            return {}

        self._set_ready_to_action(state)
        return {
            "resources" : world.resources,
            "main": {
                "team": {
                    "base": {
                        "load": world.main_team.base.load,
                        "trained_unit": False
                    },
                    "units": {
                        "all": [unit for unit in world.main_team.units],
                        "trucks": [unit for unit in world.main_team.units if unit.tag == Unit.TAGS.TRUCK],
                        "heavy_tanks": [unit for unit in world.main_team.units if unit.tag == Unit.TAGS.HEAVY_TANK],
                        "light_tanks": [unit for unit in world.main_team.units if unit.tag == Unit.TAGS.LIGHT_TANK],
                        "drones": [unit for unit in world.main_team.units if unit.tag == Unit.TAGS.DRONE],
                    },
                    "killed_unit_count": 0,
                    "died_unit_count": 0,
                }
            },
            "opponent": {
                "team": {
                    "base": {
                        "load": world.opponent_team.base.load,
                        "trained_unit": False
                    },
                    "units": {
                        "all": [unit for unit in world.opponent_team.units],
                        "trucks": [unit for unit in world.opponent_team.units if unit.tag == Unit.TAGS.TRUCK],
                        "heavy_tanks": [unit for unit in world.opponent_team.units if unit.tag == Unit.TAGS.HEAVY_TANK],
                        "light_tanks": [unit for unit in world.opponent_team.units if unit.tag == Unit.TAGS.LIGHT_TANK],
                        "drones": [unit for unit in world.opponent_team.units if unit.tag == Unit.TAGS.DRONE],
                    },
                    "killed_unit_count": 0,
                    "died_unit_count": 0,
                }
            }
        }

    def _generate_world_config(self) -> Dict[str, Any]:
        import os
        import yaml
        height, width = self._game.map_y, self._game.map_x
        map_name = getattr(self._kwargs, 'map', 'Random')
        if map_name != 'Random':
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'config', f'{map_name}.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self._game.config = config
            else:
                # Dosya yoksa random harita oluştur
                self._game.config = World.generate_random_world_config(height, width)
        else:
            self._game.config = World.generate_random_world_config(height, width)
        return self._game.reset()

    def _create_unit(self, entity: Dict[str, Any]) -> Unit:
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

    def _find_unit(self, location: Dict[int, int]) -> Optional["Unit"]:
        world = self._world
        if not world:
            return None

        units: List["Unit"] = world.main_team.units + world.opponent_team.units
        for unit in units:
            if unit.going_tile is not None:
                if unit.going_tile.location == location:
                    return unit
            else:
                if unit.location == location:
                    return unit

        return None

    def _decode_state(self, observation: Dict[str, Any]):
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

    def _action(self, estimations: List[List[int]], recruitment: int) -> Tuple[List[Tuple[int, int]], List[int], List[Tuple[int, int]], int]:
        world = self._world
        if not world:
            return ([], [], [], 0)

        actions: List[Tuple[Tuple[int, int], int, Optional[Tuple[int, int]]]] = []

        main_team = world.main_team
        opponent_team = world.opponent_team
        main_team_units = main_team.units
        for i, unit in enumerate(main_team_units):
            estimation = estimations[i]
            movement = estimation[0]
            targeting = estimation[1]
            response = unit.action(movement, targeting)
            if not response:
                continue

            source_location, direction, target_location = response

            unit.idle = False
            if not unit.tile:
                continue

            # unit.going_tile = unit.tile.neighbors[direction] if 0 < direction else unit.tile
            actions.append((source_location, direction, target_location))

        if self.action_length <= len(world.main_team.units):
            recruitment = 0

        assert len(actions) <= self.action_length
        source_locations = [action[0] for action in actions]
        directions = [action[1] for action in actions]
        target_directions = [action[2] for action in actions]
        return (source_locations, directions, target_directions, recruitment) # type: ignore

    def _flat_state_2(self, state: Dict[str, Any]) -> List[int]:
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

    def _build_map(self, state: Dict[str, Any], force: bool = False):
        if not force and self._is_map_built:
            return

        world = self._world
        if not world:
            return

        decoded_state = self._decode_state(state)
        world.set_terrain(state["terrain"], decoded_state[4])
        main_team = world.main_team
        main_team.base = Base()
        base_y, base_x = decoded_state[main_team.index + 2]
        main_team.base.tile = world.terrain.tiles[base_y][base_x]
        main_team.base.tile.has_base = True # type: ignore
        opponent_team = world.opponent_team
        opponent_team.base = Base()
        base_y, base_x = decoded_state[opponent_team.index + 2]
        opponent_team.base.tile = world.terrain.tiles[base_y][base_x]
        opponent_team.base.tile.has_base = True # type: ignore
        self._is_map_built = True

    def _set_ready_to_action(self, state: Dict[str, Any]):
        world = self._world
        decoded_state = self._decode_state(state)
        scores = state["score"]
        if not world:
            return

        world.clear()
        world.terrain.set_resource_tiles(decoded_state[4])
        # Main Team
        main_team = world.main_team
        for i, entity in enumerate(decoded_state[main_team.index]):
            [_, _, _, location, _] = entity.values()
            y, x = location
            tile = world.terrain.tiles[y][x]
            unit = self._create_unit(entity)
            unit.tile = tile
            main_team.units.append(unit)

        # Oppenent Team
        opponent_team = world.opponent_team
        for i, entity in enumerate(decoded_state[opponent_team.index]):
            [_, _, _, location, _] = entity.values()
            y, x = location
            tile = world.terrain.tiles[y][x]
            unit = self._create_unit(entity)
            unit.tile = tile
            opponent_team.units.append(unit)

    def action(self, observation: Dict[str, Any]):
        raise NotImplementedError()

class TrainAgentEnv(Agent, Env):

    def action(self, observation: Dict[str, Any]):
        # Bu fonksiyon RL ortamı için kullanılmaz, sadece Game içindeki step için gereklidir.
        # Örnek olarak tüm birimler için hareketsiz (do_nothing) aksiyon döndürülüyor.
        self._build_map(observation)
        self._set_ready_to_action(observation)
        world = self._world
        locations, movements, targets = [], [], []
        if not world:
            return (locations, movements, targets, 0)
        main_team = world.main_team
        for unit in main_team.units:
            if not unit.tile:
                continue
            locations.append(unit.location)
            movements.append(0)  # hareketsiz
            targets.append(unit.location)
        return (locations, movements, targets, 0)
    def __init__(self, kwargs: Namespace, agents: List[str]) -> None:
        super().__init__(kwargs=kwargs, agents=agents)
        self.__previous_state: Dict[str, Any] = {}
        self.__reward: float = 0.0
        self.__state: Dict[str, Any] = {}
        height, width =  self._game.map_y, self._game.map_x
        self._world = World(0, 1)
        length = height * width
        self.observation_space = Box(
            low=-32,
            high=512,
            shape=(length * 6 + 4,),
            dtype=np.int16
        )

        self.action_space = MultiDiscrete(
            [7, 7] * 7 + [5]
        )

    def setup(self, observation_space: Box, action_space: MultiDiscrete):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[List[int], Dict[str, Any]]: # type: ignore
        super().reset(seed=seed, options=options)
        self.__state = self._generate_world_config()
        self._build_map(self.__state, True)
        return self._flat_state_2(self.__state), {}

    def step(self, actions: npt.NDArray[np.int8]) -> Tuple[List[int], float, bool, bool, Dict[str, Any]]: # type: ignore
        self._build_map(self.__state)
        self.__previous_state = self.__state
        previous_world: Dict[str, Any] = self._world_to_dict(self.__previous_state)
        estimations: List[List[int]] = np.array([actions[i:i+2] for i in range(0, len(actions[:-1]), 2)], dtype=np.int8).tolist()
        recruitment = actions[-1]
        self.__state, _, done = self._game.step(self._action(estimations, recruitment))
        current_world: Dict[str, Any] = self._world_to_dict(self.__state)

        # --- Yeni ödül sistemi ---
        reward = 0.0

        # === Kaynak değişimi ===
        prev_res = previous_world["main"]["team"]["base"]["load"]
        curr_res = current_world["main"]["team"]["base"]["load"]

        reward += (curr_res - prev_res) * 0.1  # kaynak başına küçük ödül

        # === HP değişimleri ===
        def hp_dict(units):
            return {(u.location, u.tag): u.hp for u in units if u.location}

        prev_main = hp_dict(previous_world["main"]["team"]["units"]["all"])
        curr_main = hp_dict(current_world["main"]["team"]["units"]["all"])
        prev_opp  = hp_dict(previous_world["opponent"]["team"]["units"]["all"])
        curr_opp  = hp_dict(current_world["opponent"]["team"]["units"]["all"])

        # Rakibe verilen hasar
        opp_damage = sum(
            max(0, prev_hp - curr_opp.get(key, 0))
            for key, prev_hp in prev_opp.items()
        )

        # Kendi alınan hasar
        main_damage = sum(
            max(0, prev_hp - curr_main.get(key, 0))
            for key, prev_hp in prev_main.items()
        )

        reward += opp_damage * 0.5
        reward -= main_damage * 0.5

        # === Unit ölümü ===
        opp_deaths = sum(1 for key in prev_opp if key not in curr_opp)
        main_deaths = sum(1 for key in prev_main if key not in curr_main)

        reward += opp_deaths * 2.0    # düşman ölürse ekstra ödül
        reward -= main_deaths * 2.0   # kendi ölürse ekstra ceza

        return self._flat_state_2(self.__state), reward, done, False, {}

    def render(self,):
        return None

    def close(self,):
        return None


class RandomAgent(Agent):
    def __init__(self, index: int, action_length: int = ACTION_LENGTH):
        super().__init__(kwargs=Namespace(), agents=[])
        self.action_length = ACTION_LENGTH
        self._world = World(index, 1 - index)

    def action(self, observation: Dict[str, Any]):
        self._build_map(observation)
        self._set_ready_to_action(observation)
        world = self._world
        locations, movements, targets, units, = [], [], [], []
        if not world:
            return (locations, movements, targets, 0)

        blue_team = world.main_team
        red_team = world.opponent_team
        lower_bound = min(len(blue_team.units), self.action_length)
        for i in range(lower_bound):
            blue_unit = blue_team.units[i]
            if not blue_unit.tile:
                continue

            direction = random.choice(list(blue_unit.tile.neighbors.keys()))
            target = blue_unit.location
            if not target:
                continue

            (Y, X) = target
            if 0 < direction and world:
                world.reserved_tiles.append(world.terrain.tiles[Y][X])

            blue_unit.going_tile = blue_unit.tile.neighbors[direction]
            blue_unit.idle = False
            locations.append(blue_unit.location)
            units.append(blue_unit)
            movements.append(direction)
            targets.append(target)

        return (locations, movements, targets, random.randint(0, 4))



class SmartAgent(Agent):
    """An example agent which shares the same learning infrastructure as
    the full-fledged benchmarks, but implements random action selection."""

    # Private:
    __a2c: A2C

    # Public:
    observation_space: Box
    action_space: MultiDiscrete

    def __init__(self, observation_space_or_team_index: Any, action_space_or_action_length: Any):
        super().__init__(kwargs=Namespace(map="TrainSingleTruckSmall", render=True, gif=True, img=True), agents=[])
        if isinstance(observation_space_or_team_index, int):
            main_team_index = observation_space_or_team_index
            opponent_team_index = 1 - observation_space_or_team_index
            self._world = World(main_team_index, opponent_team_index)
        else:
            self._world = World(0, 1)

        self.__a2c = A2C.load(r"models\RiskyValley.zip")
        self.observation_space = self.__a2c.observation_space # type: ignore
        self.action_space = self.__a2c.action_space # type: ignore

    def action(self, observation: Dict[str, Any]):
        return SmartAgent.act(self, observation)

    def act(self, state: Dict[str, Any]):
        Agent._build_map(self, state)
        Agent._set_ready_to_action(self, state)
        actions, *_ = self.__a2c.predict(self._flat_state_2(state)) # type: ignore
        estimations: List[List[int]] = np.array([actions[i:i+2] for i in range(0, len(actions[:-1]), 2)], dtype=np.int8).tolist()
        recruitment = actions[-1]
        return self._action(estimations, recruitment)