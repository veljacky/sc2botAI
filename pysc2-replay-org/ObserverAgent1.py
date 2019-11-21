#!/usr/bin/env python
from typing import Optional, Dict, List, Any, Union

import numpy as np
import math

from numpy.core._multiarray_umath import ndarray
from pysc2.lib import static_data
import cv2
from enum import Enum
from protoss import ProtossBuildings, ProtossUnits, BUILDINGS_CODE, UNITS_CODE, ATTACK_CODE

units = [0 for i in range(18)]
buildings = [0 for i in range(13)]
buildings[0] = 1
units[0] = 12


class ObserverAgent:
    def __init__(self):
        self.states = []
        image = np.zeros((64, 64, 3))

        # Starting with 1 Nexus and 12 Probes

        self.gameloop = 0
        self.cameraXY = None


    def step(self, obs, actions):

        if obs.observation["map_name"] != "Tryton ER":
            raise Exception("Error[333] Map should be Tryton ER!")

        self.gameloop += 1
        current_state = {}

        minimap = obs.observation["feature_minimap"]                # Minimap observations
        player_obs = obs.observation["player"]                      # Structured available data
        nonzeros = np.nonzero(np.array(minimap[3]))
        self.cameraXY = (nonzeros[0][0], nonzeros[0][1])            # Current camera position


        # Image info
        current_state["minimap"] = [np.array(minimap[0]) / 255,               # Minimap height info
                                    np.array(minimap[1]) / 2,                 # Minimap visibility info
                                    np.array(minimap[2]),                     # Minimap creep location info
                                    np.array(minimap[3]),                     # Minimap camera info
                                    np.array((minimap[5] == 1).astype(int)),  # Minimap owned units
                                    np.array((minimap[5] == 4).astype(int)),  # Minimap enemy units
                                    np.array(minimap[6])]                     # Minimap selection info

        # Actions info
        if actions:
            current_state["gameloop"] = self.gameloop
            current_state["units"] = np.array(units)
            current_state["building"] = np.array(buildings)
            current_state["player"] = player_obs
            print(units)
            current_action = self.encode_action(actions, verbose=True)
            current_state["action"] = current_action
            self.states.append(current_state)

        # else:
        #     current_action = {"action_id": 0, "target": 0}
        #     print(current_action)
        #     current_state["action"] = current_action
        #     self.states.append(current_state)

    def encode_action(self, action_pysc, verbose=False):
        """ action: pysc2 style action
            returns decoded for NN action [action_id, second_action_id [first coord1, first coord2], [second_coord1, second_coord2]]
            -1 value means label is unused"""
        encoded = {"action_id": 0, "target": 0}
        target = None
        if action_pysc:

            action, args = action_pysc
            action_id = int(action)
            action_str = str(action)

            if 11 < action_id < 19:
                if "screen" in action_str:
                    target = [int(self.cameraXY[0] + args[1][0]/10.5), int(self.cameraXY[1] + args[1][0]/10.5)]
                elif "minimap" in action_str:
                    target = args[1]
                if target:
                    encoded["action_id"] = ATTACK_CODE
                    encoded["target"] = target
                else:
                    encoded["action_id"] = 0
                    encoded["target"] = 0
                return encoded

            elif 38 < action_id < 102:
                for i, building in enumerate(ProtossBuildings):
                    if building in action_str:
                        buildings[i] += 1
                        if "screen" in action_str:
                            target = [int(self.cameraXY[0] + args[1][0] / 10.5),
                                      int(self.cameraXY[1] + args[1][0] / 10.5)]
                            encoded["action_id"] = BUILDINGS_CODE + i
                            encoded["target"] = target
                        else:
                            encoded["action_id"] = 0
                            encoded["target"] = 0
                        if verbose:
                            print("Building {}: {}".format(i, building))
                            print(args)
                        break
                return encoded

            # elif 350 < action_id < 451:
            #     print("Action research!")

            elif 456 < action_id < 505:
                for i, unit in enumerate(ProtossUnits):
                    if unit in action_str:
                        units[i] += 1
                        if "quick" in action_str:
                            encoded["action_id"] = UNITS_CODE + i
                            encoded["target"] = 0
                        else:
                            encoded["action_id"] = 0
                            encoded["target"] = 0
                        if verbose:
                            print("Unit {}: {}".format(i, unit))
                        break
                return encoded
            else:
                return encoded
        else:
            return encoded
