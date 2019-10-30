import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

class MyAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MyAgent, self).__init__()

    def step(self, obs):
        if obs.first():
            print("Obserwacje!")
            for i, elem in enumerate(obs.observation):
                print("Obserwacja {}:".format(i))
                print(elem)
        return actions.RAW_FUNCTIONS.no_op()

def main(unused_argv):
    agent = MyAgent()
    try:
        while True:
            with sc2_env.SC2Env(map_name="Simple64",
                                players=[sc2_env.Agent(sc2_env.Race.protoss),
                                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    action_space=actions.ActionSpace.RAW,
                                    use_raw_units=True,
                                    raw_resolution=64)) as env:
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)