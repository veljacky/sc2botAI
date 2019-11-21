#!/usr/bin/env python
import numpy as np
import math
from pysc2.lib import static_data
import cv2
from enum import Enum
from protoss import ProtossBuildings, ProtossUnits


def encode_action(action_pysc, verbose=False):
    """ action: pysc2 style action
        returns decoded for NN action [action_id, second_action_id [first coord1, first coord2], [second_coord1, second_coord2]]
        -1 value means unused label is unused"""
    encoded = [0, 0, 0, 0, 0, 0]
    if action_pysc:
        action, args = action_pysc
        action_id = int(action)
        action_str = str(action)
        if 11 < action_id < 19:
            if "screen" in action_str:
                print("Attack on screen!")
            elif "minimap" in action_str:
                print("Attack on minimap!")
        elif 38 < action_id < 102:
            for i, building in enumerate(ProtossBuildings):
                if building in action_str:
                    print("Building {}: {}".format(i, building))
                    break
        elif 350 < action_id < 451:
            print("Action research!")
        elif 456 < action_id < 505:
            for i, unit in enumerate(ProtossUnits):
                if unit in action_str:
                    print("Unit {}: {}".format(i, unit))
                    break

    argscount = len(args)



    # if argscount == 1:
    #     if len(args[0]) == 1:
    #         encoded[1] = int(args[0][0]) +1
    #     elif len(args[0]) == 2:
    #         encoded[2] = int(args[0][0])+1
    #         encoded[3] = int(args[0][1])+1
    #     else:
    #         print(action_pysc)
    #         raise Exception("Error[2]: Unknown args pattern!")
    # elif argscount == 2:
    #     if len(args[0]) == 1:
    #         encoded[1] = int(args[0][0]) +1
    #     elif len(args[0]) == 2:
    #         encoded[2] = int(args[0][0])+1
    #         encoded[3] = int(args[0][1])+1
    #     else:
    #         print(action_pysc)
    #         raise Exception("Error[2]: Unknown args pattern!")
    #
    #     if len(args[1]) == 2:
    #         encoded[4] = int(args[1][0])+1
    #         encoded[5] = int(args[1][1])+1
    #     elif len(args[1]) == 1:
    #         encoded[2] = int(args[1][0])+1
    #     else:
    #         print(action_pysc)
    #         raise Exception("Error[2]: Unknown args pattern!")
    # elif argscount == 3:
    #     if len(args[0]) == 1:
    #         encoded[1] = int(args[0][0]) +1
    #     else:
    #         print(action_pysc)
    #         raise Exception("Error[2]: Unknown args pattern!")
    #
    #     if len(args[1]) == 2:
    #         encoded[4] = int(args[1][0])+1
    #         encoded[5] = int(args[1][1])+1
    #     else:
    #         print(action_pysc)
    #         raise Exception("Error[2]: Unknown args pattern!")
    #
    #     if len(args[2]) == 2:
    #         encoded[4] = int(args[1][0])+1
    #         encoded[5] = int(args[1][1])+1
    #     else:
    #         print(action_pysc)
    #         raise Exception("Error[2]: Unknown args pattern!")
    # if verbose:
    #     print("Action: {}, Args = {}, Ilość argumentów: {}, Encoded: {}".format(action, args, argscount, encoded))
    #     for elem in encoded:
    #         if type(elem) != int:
    #             print("Błąd przy kodowaniu!")
    return encoded

class ObserverAgent:
    def __init__(self):
        self.states = []
        image = np.zeros((64, 64, 3))

    def step(self, obs, actions):
        current_state = {}
        minimap = obs.observation["feature_minimap"]                # minimap observations
        current_state["minimap"] = [minimap[0] / 255,               # Minimap height info
                                    minimap[1] / 2,                 # Minimap visibility info
                                    minimap[2],                     # Minimap creep location info
                                    minimap[3],                     # Minimap camera info
                                    (minimap[5] == 1).astype(int),  # Minimap owned units
                                    (minimap[5] == 4).astype(int),  # Minimap enemy units
                                    minimap[6]]                     # Minimap selection info
        if obs.observation["map_name"] != "Tryton ER":
            raise Exception("Error[333] Map should be Tryton ER!")

        #
        # screen = obs.observation["feature_screen"]
        # unit_type = screen[6]
        # unit_type_compressed = np.zeros(unit_type.shape, dtype=np.float)
        # for y in range(len(unit_type)):
        #     for x in range(len(unit_type[y])):
        #         if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
        #             unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)
        #
        # hit_points = screen[8]
        # hit_points_logged = np.zeros(hit_points.shape, dtype=np.float)
        # for y in range(len(hit_points)):
        #     for x in range(len(hit_points[y])):
        #         if hit_points[y][x] > 0:
        #             hit_points_logged[y][x] = math.log(hit_points[y][x]) / 4
        #
        # current_state["screen"] = [screen[0] / 255,                 # Screen terrain height info
        #                            screen[1] / 2,                   # Screen visibility info
        #                            screen[2],
        #                            screen[3],
        #                            (screen[5] == 1).astype(int),
        #                            (screen[5] == 3).astype(int),
        #                            (screen[5] == 4).astype(int),
        #                            unit_type_compressed,
        #                            screen[7],
        #                            hit_points_logged,
        #                            screen[9]/255,
        #                            screen[10]/255]

        if actions:
            current_state["action"] = actions
            encode_action(actions, verbose=True)
            self.states.append(current_state)
