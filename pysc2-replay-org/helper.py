from pysc2.lib import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import multiprocessing
import importlib
import glob
import sys
import os

cpusNumber = multiprocessing.cpu_count()

FLAGS = flags.FLAGS
flags.DEFINE_string("replays", None, "Path to directory of replays.")
flags.DEFINE_string("agent", None, "Path to observer agent.")
flags.DEFINE_integer("procs", cpusNumber, "Number of processes to run", lower_bound=1)
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("agent")
FLAGS(sys.argv)


class ReplayEnv:
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 screen_size_px=(64, 64),
                 minimap_size_px=(64, 64),
                 discount=1.,
                 step_mul=1):

        self.agent = agent
        self.discount = discount
        self.step_mul = step_mul

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(replay_file_path)
        ping = self.controller.ping()
        info = self.controller.replay_info(replay_data)
        if not self._valid_replay(info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=False, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if info.local_map_path:
            map_data = self.run_config.map_data(info.local_map_path)

        self._episode_length = info.game_duration_loops
        self._episode_steps = 0

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            return False
#   for p in info.player_info:
#       if p.player_apm < 10 or p.player_mmr < 1000:
#           # Low APM = player just standing around.
#           # Low MMR = corrupt replay or player who is weak.
#           return False
        return True

    def start(self):
        _features = features.features_from_game_info(self.controller.game_info())

        while True:
            self.controller.step(self.step_mul)
            obs = self.controller.observe()
            try:
                agent_obs = _features.transform_obs(obs)
            except:
                pass

            if obs.player_result: # Episide over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += self.step_mul

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            self.agent.step(step, obs.actions)

            if obs.player_result:
                break

            self._state = StepType.MID

def start(agent_cls, replay_path):
    replay = ReplayEnv(replay_path, agent_cls())
    replay.start()


def main(unused):

    agent_module, agent_name = FLAGS.agent.split(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    processesNo = FLAGS.procs
    replay_dir = FLAGS.replays
    os.system('cls')
    print("Uruchomiono analizator powtórek!\nAnalizowanie powtórek z lokalizacji: {0}\n"
          "Klasa obserwatora: {1}\nLiczba wykorzystywanych procesów: "
          "{2}".format(replay_dir, agent_cls, processesNo))

    replays = glob.glob(replay_dir+'\*.SC2Replay')

    if len(replays) == 0:
        sys.stderr.write("[Error] Nie znaleziono żadnych powtórek w folderze docelowym!\n")
        return
    proc = []
    p = multiprocessing.Process(target=start, args=(agent_cls, replays[0]))
    p.start()
    proc.append(p)
    p = multiprocessing.Process(target=start, args=(agent_cls, replays[1]))
    p.start()
    proc.append(p)

    for p in proc:
        p.join()
    for i in range(len(replays)):
        pass


if __name__ == "__main__":
    app.run(main)
