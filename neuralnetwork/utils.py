import numpy as np
import random
import cv2


def reduce_noop(listDS, iterations=1):
    for i in range(iterations):
        for i, row in enumerate(listDS):
            if row["action"] == {'action_id': 0, 'target': 0}:
                if random.randint(0, 2) != 0:
                    listDS.pop(i)


def action_to_cathegory(act):
    cathegory = None
    id = act["action_id"]
    if id == 0:
        cathegory = 0
    elif 99 < id < 200:
        # Recruit action
        cathegory = 1 + id % 100
    elif 199 < id < 300:
        # Build Action
        cathegory = 19 + id % 200
    elif 299 < id < 400:
        cathegory = 32 + id % 300
    return cathegory


def reduce_cathegories(act):
    cathegory = 0
    id = act
    if id == 0:
        # Noop
        cathegory = 0
    elif 0 < id < 19:
        # Recruit action
        # if id > 4:
        #     cathegory = 3
        # else:
        #     cathegory = id
        cathegory = 1
    elif 18 < id < 32:
        # if id > 24:
        #     cathegory = random.randint(1, 4)
        # elif id == 23 or id == 24:
        #     cathegory = 9
        # else:
        #     cathegory = id - 19 + 5
        cathegory = 2
    elif id == 32:
        cathegory = 3
    else:
        cathegory = 0
    return cathegory


def data_to_image(minimap, visualize=False):
    height_info = minimap[0]
    visibility = np.array((minimap[1] * 255), dtype='uint8')
    ally_units = np.array(minimap[4], dtype='uint8')
    enemy_units = np.array(minimap[5], dtype='uint8')

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
    visualization = visualization / 255                        # Normalize to [0,1]
    if visualize:
        vis = cv2.resize(visualization, (512, 512), cv2.INTER_NEAREST)
        cv2.imshow("Visualization", vis)
        cv2.waitKey(1)
    return visualization
