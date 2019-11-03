import glob
import mpyq
import tqdm
import os
from s2protocol import versions

RIGHT_BUILD = 76811


replays = glob.glob("C:\Replays\*.SC2Replay")

counter = 0
wrong_paths = []
# for replay in tqdm.tqdm(replays, total=len(replays)):
#     archive = mpyq.MPQArchive(replay)
#     contents = archive.header['user_data_header']['content']
#     header = versions.latest().decode_replay_header(contents)
#     baseBuild = header['m_version']['m_baseBuild']
#     if baseBuild != RIGHT_BUILD:
#         counter += 1
#         wrong_paths.append(replay)
#
# for path in wrong_paths:
# #     os.remove(path)
#
# print("Znaleziono {0} niepoprawnych powtórek!".format(counter))
archive = mpyq.MPQArchive(replays[3])
contents = archive.files
protocol = versions.build(RIGHT_BUILD)

# header = versions.latest().decode_replay_header(contents)
# baseBuild = header['m_version']['m_baseBuild']
# print(header)
