import random
from argparse import Namespace
from enum import Enum

import numpy as np
from game import Game
from gym import Env
from gym.spaces import Box, MultiDiscrete
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


class Tile:
    class TAGS(Enum):
        GRASS = 0
        DIRT = 1
        MOUNTAIN = 2
        WATER = 3

    IS_MOVEABLE: bool = True
    IS_STICKY: bool = False

    def __init__(self) -> None:
        self.tag = None
        self.neighbors: dict[int, Tile] = {}
        self.has_resource = False
        self.has_base = False
        self.location: tuple[int, int] = (-1, -1)


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
        self._cost: int = -1
        self._attack: int = -1
        self._max_hp: int = -1
        self._heavy: bool = False
        self._fly: bool = False
        self._anti_air: bool = False
        self._location = location
        self._tile: Tile = None
        self.hp = hp
        self.load = load
        self.tag = tag
        self.will_move = False
        self.going_tile: Tile = None

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

    def get_available_directions(self, units: list[Self], reserved_tiles: list[Tile], blue_base_tile: Tile, red_base_tile) -> list[int]:
        available_directions: list[int] = []
        for direction, tile in self._tile.neighbors.items():
            if tile in [*[reserved_tile for reserved_tile in reserved_tiles],
                        blue_base_tile,
                        red_base_tile,
                        *[unit.tile for unit in units if not unit.will_move and unit.tile is not self._tile]]:
                continue

            if tile in [unit.going_tile for unit in units if unit.will_move]:
                continue

            if self.tag is Unit.TAGS.HEAVY_TANK and tile.IS_STICKY:
                continue

            if self.tag is not Unit.TAGS.DRONE and not tile.IS_MOVEABLE:
                continue

            available_directions.append(direction)

        return available_directions


class HeavyTankUnit(Unit):
    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self._cost: int = 2
        self._attack: int = 2
        self._max_hp: int = 4
        self._heavy: bool = True
        self._fly: bool = False
        self._anti_air: bool = False


class LightTankUnit(Unit):
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

    def is_full(self) -> bool:
        return self.load >= self._max_load

    def is_on_resource(self) -> bool:
        return self._tile.has_resource

    def is_on_base(self) -> bool:
        return self._tile.has_base

    def get_nearest_resource(self, resources: list[Tile]) -> Tile:
        return min(resources, key=lambda resource: get_distance_between_two_locations(self.location, resource.location), default=None)

    def get_available_directions(self, units: list[Self], reserved_locations: list[Tile], blue_base_tile: Tile, red_base_tile: Tile) -> list[int]:
        available_directions = super().get_available_directions(
            units, reserved_locations, blue_base_tile, red_base_tile)
        if not self.is_on_base() and not self.has_load():
            return available_directions

        for direction, tile in self._tile.neighbors.items():
            if tile.has_base and tile is blue_base_tile:
                available_directions.append(direction)
                break

        return available_directions

    def pick_up_resource(self) -> None:
        self._is_resource_picked_up = True

    def deliver_resource(self) -> None:
        self._is_resource_delivered = True


class DroneUnit(Unit):

    def __init__(self, hp: int, load: int, location: tuple[int, int], tag: Unit.TAGS) -> None:
        super().__init__(hp, load, location, tag)
        self._cost: int = 1
        self._attack: int = 1
        self._max_hp: int = 1
        self._heavy: bool = False
        self._fly: bool = True
        self._anti_air: bool = True


class Base:
    def __init__(self, location: tuple[int, int], load: int) -> None:
        self.location: tuple[int, int] = location
        self.load: int = load
        self.on_tile: Tile = None
        self.trained: bool = False

    def train(self):
        self.trained = True


class Team:
    def __init__(self, index: int) -> None:
        self.index: int = index
        self.units: list[Unit] = []
        self.base: Base = None


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


class BaseAgent():
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

    def _action(self, estimations: dict[tuple[int, int], dict[str, any]], recruitment: int):
        world = self._world
        locations, movements, targets, = [], [], []

        # needed variables
        main_team = world.main_team
        opponent_team = world.opponent_team
        main_team_units = main_team.units
        opponent_team_units = opponent_team.units
        sorted_main_team_units = sorted(main_team_units, key=lambda unit: get_distance_between_two_locations(
            main_team.base.location, unit.location))

        # running for each unit
        for unit in main_team_units:

            # checking that there are some directions to action
            available_directions = unit.get_available_directions(
                [*main_team_units, *opponent_team_units],
                world.reserved_tiles,
                world.main_team.base.on_tile,
                world.opponent_team.base.on_tile
            )
            if len(available_directions) == 0:
                continue

            estimation = estimations[unit.location]
            direction, target = estimation["direction"], estimation["target"]
            if direction != 0 and direction not in available_directions:
                continue

            sorted_opponent_units_to_unit = sorted(opponent_team.units, key=lambda unit: get_distance_between_two_locations(
                unit.location, unit.location))
            if unit.tag is Unit.TAGS.TRUCK:
                target = unit.location if direction == 0 else None
                if target is not None:
                    if unit.has_load() and unit.is_on_base():
                        unit.deliver_resource()
                    elif unit.has_space() and unit.is_on_resource():
                        unit.pick_up_resource()
                    else:
                        target = None

            if target is not None:
                if unit.tag is not Unit.TAGS.TRUCK:
                    distance_to_target = get_distance_between_two_locations(
                        unit.location, target)
                    if 2 < distance_to_target:
                        target = None

            if 0 < direction:
                world.reserved_tiles.append(unit.tile.neighbors[direction])
            else:
                if unit.tag == Unit.TAGS.TRUCK:
                    if (unit.has_space() and unit.is_on_resource()) or (unit.has_load() and unit.is_on_base()):
                        target = unit.location

            if target is None and direction == 0:
                continue

            unit.will_move = True
            unit.going_tile = unit.tile.neighbors[direction] if 0 < direction else unit.tile

            locations.append(unit.location)
            movements.append(direction)
            targets.append(target)

        assert all(len(x) <= self.action_length for x in [
                   locations, movements, targets])
        return (locations, movements, targets, recruitment)

    def _flat_state(self, state: dict[str, any]):
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
    def __init__(self):
        self._world: World
        self._state: dict[str, any]
        self.action_length: int = ACTION_LENGTH
        pass

    def _set_ready_to_action(self, state: dict[str, any]) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]], int]:
        world = self._world
        world.clear()
        decoded_state = self._decode_state(state)
        world.terrain = state["terrain"]
        world.resources = decoded_state[4]
        scores = state["score"]

        # Main Team
        main_team = self._world.main_team
        main_team.base = Base(
            decoded_state[main_team.index + 2], int(scores[main_team.index]))
        for i, entity in enumerate(decoded_state[main_team.index]):
            [_, _, _, location, _] = entity.values()
            y, x = location
            tile = world.terrain.tiles[y][x]
            unit = self._create_unit(entity)
            unit.tile = tile
            tile.unit = unit
            main_team.units.append(unit)

        base_y, base_x = main_team.base.location
        main_team.base.on_tile = world.terrain.tiles[base_y][base_x]
        world.terrain.tiles[base_y][base_x].has_base = True

        # Oppenent Team
        opponent_team = self._world.opponent_team
        opponent_team.base = Base(
            decoded_state[opponent_team.index + 2], int(scores[opponent_team.index]))
        for i, entity in enumerate(decoded_state[opponent_team.index]):
            [_, _, _, location, _] = entity.values()
            y, x = location
            tile = world.terrain.tiles[y][x]
            unit = self._create_unit(entity)
            unit.tile = tile
            tile.unit = unit
            opponent_team.units.append(unit)

        base_y, base_x = opponent_team.base.location
        opponent_team.base.on_tile = world.terrain.tiles[base_y][base_x]
        world.terrain.tiles[base_y][base_x].has_base = True

    def _update_after_action(self, state: dict[str, any], previous_state: dict[str, any]):
        decoded_state = self._decode_state(state)
        previous_decoded_state = self._decode_state(previous_state)
        world = self._world
        # world.resources = decoded_state[4]
        scores = state["score"]

        # TODO: eğer base location da yeni bir birim varsa üeritilmiş demektir. bunu listene ekleyeceğiz.
        # TODO: rakip dusmanla ayni yere gitmeye calisabilirler. cozmeliyiz

        main_team = self._world.main_team
        opponent_team = self._world.opponent_team

        main_team.base.load = int(scores[main_team.index])
        opponent_team.base.load = int(scores[opponent_team.index])

        tiles_in_row: list[Tile] = [
            tile for row in world.terrain.tiles for tile in row]
        checked_tiles: list[Tile] = []

        def find_tail_of_chain(tile: Tile) -> Tile:
            if tile.coming_unit is None:
                return tile

            return find_tail_of_chain(tile.coming_unit.tile)

        main_decoded_units = decoded_state[main_team.index]
        opponent_decoded_units = decoded_state[opponent_team.index]
        all_decoded_units = main_decoded_units = opponent_decoded_units

        for tile in tiles_in_row:
            if tile in checked_tiles:
                continue

            unit = tile.unit
            if unit is None:
                continue

            head_of_chain: Tile = unit.going_tile

            if head_of_chain is None:
                continue

            if head_of_chain.location in [opponent_unit["location"] for opponent_unit in opponent_decoded_units]:
                continue

            tail_of_chain: Tile = find_tail_of_chain(head_of_chain)
            unit_on_tail: Unit = tail_of_chain.unit
            if unit_on_tail is None:
                continue

            current_tile: Tile = head_of_chain
            while True:
                coming_unit = current_tile.coming_unit
                if coming_unit is None:
                    break

                checked_tiles.append(current_tile)
                coming_tile = coming_unit.tile
                coming_tile.unit = None
                coming_unit.tile = current_tile
                current_unit = current_tile.unit
                current_unit.going_tile = None
                current_unit.will_move = False
                current_tile.coming_unit = None
                current_tile = coming_tile

        # TODO: karşının harketini nasıl bileceğiz?
        unit_list = decoded_state[main_team.index]
        is_there_new_unit = len(main_team.units) < len(unit_list)

        for i, entity in enumerate(decoded_state[main_team.index]):
            [_, tag, hp, location, load] = entity.values()
            unit = self._find_unit(location)
            if unit is None:
                y, x = location
                unit = self._create_unit(entity)
                unit.tile = world.terrain.tiles[y][x]
                main_team.units.append(unit)
            else:
                unit.hp = hp
                unit.load = load

         # Oppenent Team
        opponent_team = self._world.opponent_team
        opponent_team.base.load = int(scores[opponent_team.index])

        previous_decoded_state = self._decode_state(previous_state)
        for i, entity in enumerate(previous_decoded_state[opponent_team.index]):
            [_, tag, hp, location, load] = entity.values()
            unit = self._find_unit(location)
            unit.hp = hp
            unit.load = load


class TrainAgentEnv(BaseAgent, Env):
    def __world_to_dict(self, state: dict[str, any]) -> dict[str, any]:
        self._set_ready_to_action(self.__state)
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



    def __init__(self, kwargs: Namespace, agents: list[str]) -> None:
        super().__init__()
        self.__previous_state: dict[str, any] = {}
        self.__state: dict[str, any] = {}
        self.__episodes: int = 0
        self.__steps: int = 0
        self.__reward: float = 0.0
        self.__game = Game(kwargs, agents)
        self._world = World(0, 1)
        height, width =  self.__game.map_y, self.__game.map_x
        self._ASL: list[int] = [7,  height, width]
        self._ASL_LENGTH: int = len(self._ASL)
        length = height * width
        print(length)
        self.observation_space = Box(
            low=-2,
            high=401,
            shape=(length * 10 + 4,),
            dtype=np.int16
        )
        self.action_space = MultiDiscrete(np.array([
            self._ASL for _ in range(length)
        ]).flatten().tolist() + [5])
        # self.action_space = MultiDiscrete(
        #     [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5])

    def setup(self, observation_space: Box, action_space: MultiDiscrete):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        self.__episodes += 1
        self.__steps = 0
        self.__state = self.__game.reset()
        self.__reward = 0.0
        return self._flat_state(self.__state)

    def step(self, sample: np.ndarray):
        self.__steps += 1
        self.__previous_state = self.__state
        previous_world: dict[str, any] = self.__world_to_dict(self.__previous_state)
        pre_estimations = [sample[i:i+self._ASL_LENGTH] for i in range(0, len(sample[:-1]), self._ASL_LENGTH)][::-1]
        estimations: dict[tuple[int,int], dict[str, any]] = {}
        for y in range(self._world.height):
            for x in range(self._world.width):
                direction, Y, X = pre_estimations.pop()
                estimations[(y, x)] = {
                    "direction": direction,
                    "target": (Y, X)
                }

        recruitment = sample[-1]
        self.__state, _, done = self.__game.step(self._action(estimations, recruitment))
        current_world: dict[str, any] = self.__world_to_dict(self.__state)

        reward = 0.0
        # Gold Collection
        for unit in previous_world["main"]["team"]["units"]["trucks"]:
            # reward += unit.load * 8
            if unit.is_resource_picked_up():
                reward += 4
            elif unit.is_resource_delivered():
                reward += 16

            reward += unit.load * 2

            if unit.has_space():
                distance_to_resource = get_distance_between_two_locations(unit.location, unit.get_nearest_resource(previous_world["resources"]).location)
                reward += (1 - distance_to_resource / self._world.length) * 0.001 * unit.load

            if unit.has_load():
                distance_to_base = get_distance_between_two_locations(unit.location, self._world.main_team.base.location)
                reward += (1 - distance_to_base / self._world.length) * (0.002 if unit.is_full() else 0.008) * unit.load

        # Main
        reward += previous_world["main"]["team"]["base"]["load"] * 20
        diff_between_unit_counts = len(previous_world["main"]["team"]["units"]["all"]) - len(current_world["main"]["team"]["units"]["all"])
        current_world["main"]["team"]["died_unit_count"] = abs(diff_between_unit_counts) if diff_between_unit_counts < 0 else 0
        current_world["main"]["team"]["base"]["trained_unit"] = 0 < diff_between_unit_counts

        if current_world["main"]["team"]["base"]["trained_unit"]:
            reward += 4
        elif 0 < current_world["main"]["team"]["died_unit_count"]:
            reward -= 8

        # for unit in current_world["main"]["team"]["units"]["all"]:
        #     reward -= unit.max_hp - unit.hp
        #     if not unit.will_move:
        #         reward -= 5

        # Opponent
        reward -= previous_world["opponent"]["team"]["base"]["load"] * 40
        diff_between_unit_counts = len(previous_world["opponent"]["team"]["units"]["all"]) - len(current_world["opponent"]["team"]["units"]["all"])
        current_world["opponent"]["team"]["died_unit_count"] = abs(diff_between_unit_counts) if diff_between_unit_counts < 0 else 0
        current_world["opponent"]["team"]["base"]["trained_unit"] = 0 < diff_between_unit_counts

        if current_world["opponent"]["team"]["base"]["trained_unit"]:
            reward -= 4
        elif 0 < current_world["opponent"]["team"]["died_unit_count"]:
            reward += 8

        for unit in current_world["opponent"]["team"]["units"]["all"]:
            reward += unit.max_hp - unit.hp

        # if unit.is_full():
        #     reward -= 4

        #     if unit.has_space():
        #         distance_to_resource = get_distance_between_two_locations(unit.location,
        #                                                                   unit.get_nearest_resource(world.resources).location)
        #         reward +=( world.length - distance_to_resource )* 1.5

        # if unit.has_load():
        #     distance_to_base = get_distance_between_two_locations(unit.location,
        #                                                           main_team.base.location)
        #     reward += (world.length - distance_to_base) / world.length

        # reward -= len([unit for unit in main_team_units if not unit.will_move]) * 2000
        #     if truckUnit.has_space():
        #         nearest_resource = truckUnit.get_nearest_resource(
        #             world.resources)
        #         if nearest_resource:
        #             distance = get_distance_between_two_locations(
        #                 truckUnit.location, nearest_resource.location)
        #             reward += (1 - distance /
        #                        world.distance_between_two_corner) ** 0.8
        #         else:
        #             reward += 2

        #     if truckUnit.has_load():
        #         reward += truckUnit.load * 2
        #         reward += (1 - get_distance_between_two_locations(truckUnit.location,
        #                    blue_team.base.location) / world.distance_between_two_corner) ** 0.4

        # for unit in blueUnits:
        #     if not unit.will_move:
        #         reward -= 0.8
        # reward += sum([unit.load for unit in truckUnits])
        # reward += sum([unit.load * (0.02 ** (unit.max_load - unit.load))
        #               for unit in truckUnits if unit.is_on_resource(self._world.resources)])
        # reward += sum([unit.load for unit in truckUnits if unit.has_load()
        #               and unit.is_on_base(self._world)]) * 0.08
        # reward -= sum([1 for unit in truckUnits if unit.has_load() and (
        #     not unit.is_on_base(self._world.blue_team.base) or not unit.is_on_resource(self._world.resources))]) * 0.04

        #
        # reward -= sum([1 for unit in self._world.blue_team.units if unit.will_move]) * 0.008

        # reward -= len([unit for unit in self._world.blue_team.units if not unit.will_move]) * 2
        # for truck in [unit for unit in self._world.blue_team.units if unit.tag == UNIT_TAGS.TRUCK]:
        #     if truck.has_load():
        #         reward -= self._world.get_distance_between_two_locations(
        #             truck.location, self._world.blue_team.base.location) * 16
        #         if truck.is_on_base(self._world):
        #             reward -= 256

        # for location, direction, target in zip(*action[:-1]):
        #     unit = next(
        #         unit for unit in self._world.blue_team.units if unit.location == location)
        #     if unit.tag != UNIT_TAGS.TRUCK:
        #         if direction == 0 and target == None:
        #             reward -= 8
        #         else:
        #             reward += 16
        #     else:
        #         if direction == 0 and target != location:
        #             reward -= 16
        #         else:
        #             reward += 32

        # reward -= self._world.get_distance_between_two_locations(
        #     self._world.blue_team.base.location, None) ** 2
        # if len(self._world.red_team.units) == 0:
        #     reward += 4096

        # self.__reward += reward
        self.__reward += reward
        print(f"{self.__reward:0.2f}")
        return self._flat_state(self.__previous_state), self.__reward, done, {}

    def render(self,):
        return None

    def close(self,):
        return None


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
        # self.observation_space = observation_space
        # self.action_space = action_space

        self._world = World(0, 1)
        self.__a2c = A2C.load(
            ".\\models\\TrainSingleTruckLarge\\test_5040000_steps")
        self.observation_space = self.__a2c.observation_space
        self.action_space = self.__a2c.action_space

    def action(self, observation: dict[str, any]):
        return self.act(observation)

    def act(self, observation: dict[str, any]):
        sample, *_ = self.__a2c.predict(self._flat_state(observation))
        print(sample)
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
        world = self._world
        locations, movements, targets, units, = [], [], [], []
        blue_team = world.main_team
        red_team = world.opponent_team
        lower_bound = min(len(blue_team.units), self.action_length)
        for i in range(lower_bound):
            blue_unit = blue_team.units[i]
            available_directions = blue_unit.get_available_directions(
                [*blue_team.units, *red_team.units],
                world.reserved_tiles,
                world.main_team.base.on_tile,
                world.opponent_team.base.on_tile
            )
            if len(available_directions) == 0:
                continue

            direction = random.choice(available_directions)
            target = blue_unit.location
            (Y, X) = target
            if 0 < direction:
                world.reserved_tiles.append(world.terrain.tiles[Y][X])

            blue_unit.going_tile = blue_unit.tile.neighbors[direction]
            blue_unit.will_move = True
            locations.append(blue_unit.location)
            units.append(blue_unit)
            movements.append(direction)
            targets.append(target)

        assert all(len(x) <= self.action_length for x in [
                   locations, movements, targets])
        return (locations, movements, targets, random.randint(0, 4))
