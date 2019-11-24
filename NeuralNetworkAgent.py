import random
import numpy as np
import cv2
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from keras.models import load_model

operations = ["Noop!", "Recruit!", "Build!", "Attack!"]
build_order = ["pylon", "pylon", "pylon"]
recruit_order = ["probe", "probe", "probe"]

class MyAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MyAgent, self).__init__()
        self.base_location_TL = None
        self.actionmanager = load_model("neuralnetwork\\\models\\best_weights.hdf5")
        self.units = [0 for i in range(18)]
        self.buildings = [0 for i in range(13)]
        self.buildings[0] = 1
        self.units[0] = 12
        self.ready2perform = False
        self.visualization = False
        self.hold = False
        self.queued = 0
        self.actionToPerform = 0
        self.progress = 0
        self.tryCounter = 0
        self.buildingsQueued = None
        self.startLOC = None
        self.minerals = 0

    def step(self, obs):
        super(MyAgent, self).step(obs)

        if obs.first():
            print("Start!")
            nexus = self.get_units_by_type(obs, units.Protoss.Nexus)[0]
            print("Nexus LOC: X = {} Y = {}".format(nexus.x, nexus.y))
            if nexus.y > 42:
                print("Starting position: Right Bottom")
                self.startLOC = 1
            else:
                self.startLOC = 0
                print("Starting position: Left Top")

        self.minerals = int(obs.observation["player"][1])

        if obs.observation["game_loop"] % 10 == 0 and not self.hold:
            self.actionToPerform = self.make_decision(obs)

        return self.executeOrder(obs)

    def executeOrder(self, obs):
        if self.actionToPerform != 0:
            if not self.hold:
                self.hold = True

            if self.actionToPerform == 1:
                print("Recruit!")
                return self.recruit(obs)

            elif self.actionToPerform == 2:
                return self.build(obs)

            elif self.actionToPerform == 3:
                print("Attack!")
                pass

            return actions.RAW_FUNCTIONS.no_op()

        else:
            return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type]

    def build(self, obs):
        print("Build!")
        if build_order and self.buildingsQueued is None:
            building = build_order.pop()
            if building == "pylon":
                return self.buildPylon(obs)
        elif self.buildingsQueued == "pylon":
            return self.buildPylon(obs)
        return actions.RAW_FUNCTIONS.no_op()

    def buildPylon(self, obs):
        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        if len(probes) > 0 and self.minerals >= 100:

            if self.startLOC == 1:
                pylon_xy = (45, 45)
            else:
                pylon_xy = (19, 19)
            print("Build pylon! Loc: {}".format(pylon_xy))
            dists = self.get_distances(probes, pylon_xy)
            probe = probes[np.argmin(dists)]
            self.hold = False
            self.buildingsQueued = None
            self.buildings[1] += 1
            return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
        else:
            self.buildingsQueued = "pylon"
            return actions.RAW_FUNCTIONS.no_op()

    def get_distances(self, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def recruit(self, obs):
        return self.recruit_probe(obs)

    def recruit_probe(self, obs):
        nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
        if nexus:
            nexus = nexus[0]
            self.units[0] += 1
            self.hold = False
            return actions.RAW_FUNCTIONS.Train_Probe_quick("now", nexus.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def make_decision(self, obs):
        state = self.get_nn_data(obs)
        predictedaction = np.argmax(self.actionmanager.predict(state))

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
    agent = MyAgent()
    try:
        while True:
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
