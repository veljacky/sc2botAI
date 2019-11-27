import random
import numpy as np
import cv2
import json
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from keras.models import load_model
from math import sqrt
from operator import itemgetter

recruit_order = ["probe", "probe", "probe"]


def load_build_order(filename):
    with open(filename) as f:
        build_order = json.load(f)
    return build_order


class MyAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MyAgent, self).__init__()
        self.actionmanager = load_model("neuralnetwork\\\models\\best_weights.hdf5")
        self.units = [0 for i in range(18)]
        self.buildings = [0 for i in range(13)]
        self.buildings[0] = 1
        self.units[0] = 12
        self.visualization = False
        self.hold = False
        self.actionToPerform = 0
        self.buildingsQueued = False
        self.minerals = 0
        self.vespane = 0
        self.buildOrder = None
        self.recruitOrder = None
        self.nexusCoords = (None, None)
        self.markTag = None

    def step(self, obs):
        super(MyAgent, self).step(obs)

        if obs.first():
            print("Start!")
            nexus = self.get_units_by_type(obs, units.Protoss.Nexus)[0]
            print("Nexus LOC: X = {} Y = {}".format(nexus.x, nexus.y))
            self.nexusCoords = (nexus.x, nexus.y)
            if nexus.y > 42:
                print("Starting position: Right Bottom")
                order = load_build_order("build_orderRB.json")
                self.buildOrder = order["buildings"]
                self.recruitOrder = order["units"]

            else:
                print("Starting position: Left Top")
                order = load_build_order("build_orderLT.json")
                self.buildOrder = order["buildings"]
                self.recruitOrder = order["units"]

        self.minerals = int(obs.observation["player"][1])
        self.vespane = int(obs.observation["player"][2])

        # if self.markTag is not None and obs.observation["game_loop"] % 100 == 0:
        #     unit = [unit for unit in obs.observation.raw_units
        #             if unit.tag == self.markTag]
        #     print("Wskaznik coords XY : {}".format((unit[0].x, unit[0].y)))

        if obs.observation["game_loop"] % 10 == 0 and not self.hold:
            self.actionToPerform = self.make_decision(obs)

        return self.executeOrder(obs)

    def executeOrder(self, obs):
        if self.actionToPerform != 0:
            if not self.hold:
                self.hold = True

            if self.actionToPerform == 1:
                return self.recruit(obs)

            elif self.actionToPerform == 2:
                return self.build(obs)

            elif self.actionToPerform == 3:
                self.hold = False

            return actions.RAW_FUNCTIONS.no_op()

        else:
            return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type]

    def build(self, obs):
        if self.buildOrder:
            building = self.buildOrder[0]["type"]
            print("I want to build {} !".format(building))
            if building == "pylon":
                return self.buildPylon(obs)
            elif building == "gateway":
                return self.buildGateway(obs)
            elif building == "assimilator":
                return self.buildAssimilator(obs)
            elif building == "cyberneticscore":
                return self.buildCyberneticsCore(obs)
        elif not self.buildOrder:
            self.actionToPerform = 0
            self.hold = False
            return actions.RAW_FUNCTIONS.no_op()

    def buildPylon(self, obs):
        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        if len(probes) > 0 and self.minerals >= 100:
            building = self.buildOrder.pop(0)
            pylon_xy = (building["x"], building["y"])
            print("Build pylon! Loc: {}".format(pylon_xy))
            dists = self.get_distances(probes, pylon_xy)
            probe = probes[np.argmin(dists)]
            self.markTag = probe.tag
            self.hold = False
            self.buildingsQueued = False
            self.buildings[1] += 1
            self.actionToPerform = 0
            return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
        else:
            self.buildingsQueued = True
            return actions.RAW_FUNCTIONS.no_op()

    def buildGateway(self, obs):
        probes = self.get_units_by_type(obs, units.Protoss.Probe)

        if len(probes) > 0 and self.minerals >= 150 and len(self.get_completed_units(obs, units.Protoss.Pylon)) > 0:
            building = self.buildOrder.pop(0)
            gateway_xy = (building["x"], building["y"])

            print("Minerals: {} Build Gateway! Loc: {}".format(self.minerals, gateway_xy))
            dists = self.get_distances(probes, gateway_xy)
            probe = probes[np.argmin(dists)]
            self.hold = False
            self.buildingsQueued = False
            self.buildings[3] += 1
            self.actionToPerform = 0
            return actions.RAW_FUNCTIONS.Build_Gateway_pt("now", probe.tag, gateway_xy)
        else:
            self.buildingsQueued = True
            return actions.RAW_FUNCTIONS.no_op()

    def buildAssimilator(self, obs):
        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        if len(probes) > 0 and self.minerals >= 150:
            building = self.buildOrder.pop(0)
            geysers = self.get_units_by_type(obs, units.Neutral.PurifierVespeneGeyser)
            geysers.extend(self.get_units_by_type(obs, units.Neutral.RichVespeneGeyser))
            geysers.extend(self.get_units_by_type(obs, units.Neutral.ProtossVespeneGeyser))
            geysers.extend(self.get_units_by_type(obs, units.Neutral.VespeneGeyser))
            geysers.extend(self.get_units_by_type(obs, units.Neutral.ShakurasVespeneGeyser))
            geysers.extend(self.get_units_by_type(obs, units.Neutral.SpacePlatformGeyser))

            dists = self.get_distances(geysers, self.nexusCoords)
            zipped = list(zip(dists, geysers))
            zipped.sort(key=itemgetter(0))
            geyser = zipped[self.buildings[2]][1]
            print("Minerals: {} Build Assimilator! Loc: {}".format(self.minerals, (geyser.x, geyser.y)))
            dists = self.get_distances(probes, (geyser.x, geyser.y))
            probe = probes[np.argmin(dists)]

            self.hold = False
            self.buildingsQueued = False
            self.buildings[2] += 1
            self.actionToPerform = 0
            return actions.RAW_FUNCTIONS.Build_Assimilator_unit("now", probe.tag, geyser.tag)
        else:
            self.buildingsQueued = True
            return actions.RAW_FUNCTIONS.no_op()

    def buildCyberneticsCore(self, obs):
        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        if len(probes) > 0 and self.minerals >= 150 and len(self.get_completed_units(obs, units.Protoss.Gateway)) > 0:
            building = self.buildOrder.pop(0)
            cyberCoords = (building["x"], building["y"])
            print("Minerals: {} Build Cybernetics Core! Loc: {}".format(self.minerals, cyberCoords))
            dists = self.get_distances(probes, cyberCoords)
            probe = probes[np.argmin(dists)]

            self.hold = False
            self.buildingsQueued = False
            self.buildings[5] += 1
            self.actionToPerform = 0
            return actions.RAW_FUNCTIONS.Build_CyberneticsCore_pt("now", probe.tag, cyberCoords)
        else:
            self.buildingsQueued = True
            return actions.RAW_FUNCTIONS.no_op()

    def get_distances(self, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        distances = []
        for elem in units_xy:
            distance = sqrt((xy[0] - elem[0])**2 + (xy[1] - elem[1])**2)
            distances.append(distance)
        return distances

    def get_completed_units(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.SELF]

    def recruit(self, obs):
        if self.recruitOrder:
            unit_type = self.recruitOrder[0]["type"]
            if unit_type == "probe":
                return self.recruit_probe(obs)
            else:

                return actions.RAW_FUNCTIONS.no_op()
        else:
            self.hold = False
            return actions.RAW_FUNCTIONS.no_op()

    def recruit_probe(self, obs):
        nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
        if nexus and self.minerals >= 50 and (obs.observation.player[4] - obs.observation.player[3]) > 0:
            nexus = nexus[0]
            self.recruitOrder.pop(0)
            self.units[0] += 1
            self.hold = False
            self.actionToPerform = 0
            print("Recruited Probe!")
            return actions.RAW_FUNCTIONS.Train_Probe_quick("now", nexus.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def make_decision(self, obs):
        state = self.get_nn_data(obs)
        predictedaction = np.argmax(self.actionmanager.predict(state))
        print("Decision... {}".format(predictedaction))
        return predictedaction

    def get_nn_data(self, obs):
        minimap = obs.observation["feature_minimap"]
        player = obs.observation["player"]
        predictedaction = 0
        # Structured data
        observations = [obs.observation["game_loop"] / 100000]
        observations.extend(np.array(player) / 10000)
        observations.extend(np.array(self.units) / 10000)
        observations.extend(np.array(self.buildings) / 10000)

        # Visual info
        height_info = np.array(minimap[0]) / 255
        visibility = np.array((255 * minimap[1] / 2), dtype='uint8')
        ally_units = np.array((minimap[5] == 1).astype(int), dtype='uint8')
        enemy_units = np.array((minimap[5] == 4).astype(int), dtype='uint8')

        ret, visibility = cv2.threshold(visibility, 130, 255, cv2.THRESH_BINARY)
        height = cv2.bitwise_and(height_info, height_info, mask=visibility)
        vis = np.repeat(height.reshape(64, 64, 1), 3, axis=2)
        visualization = (255 * vis).astype(dtype='uint8')
        for i in range(64):
            for j in range(64):
                if ally_units[i, j] > 0:
                    visualization[i, j, :] = (0, 255, 0)
                elif enemy_units[i, j] > 0:
                    visualization[i, j, :] = (0, 0, 255)
        visualization = visualization / 255  # Normalize to [0,1]
        visualization = np.reshape(visualization, (1, 64, 64, 3))
        observations = np.reshape(observations, (1, 43))
        if self.visualization:
            vis = cv2.resize(visualization, (512, 512), cv2.INTER_NEAREST)
            cv2.imshow("Visualization", vis)
            cv2.waitKey(1)
        return [observations, visualization]


def main(unused_argv):
    try:
        while True:
            agent = MyAgent()
            with sc2_env.SC2Env(map_name="Triton",
                                players=[sc2_env.Agent(sc2_env.Race.protoss),
                                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    action_space=actions.ActionSpace.RAW,
                                    use_raw_units=True,
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True,
                                    raw_resolution=64)) as env:
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
