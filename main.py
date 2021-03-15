import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from utilities import YCBModels, Camera
import time


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)
    camera = None

    env = ClutteredPushGrasp(ycb_models, camera, vis=True)
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    env.reset()
    env.SIMULATION_STEP_DELAY = 1 / 240.
    while True:
        for _ in range(240):
            # env.step(env.read_debug_parameter(), 'end')
            env.step((0, 0, 0.2, 0, 3.14, -3.14 / 2, 0.02), 'end')
        for _ in range(240):
            env.step([0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02], 'joint')


if __name__ == '__main__':
    user_control_demo()
    # heuristic_demo()
