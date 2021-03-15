import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, setup_sisbot, setup_sisbot_force, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp:
    OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.15
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.2
    GRASP_POINT_OFFSET_Z = 1.231 - 1.1
    PUSH_POINT_OFFSET_Z = 0.0
    PUSH_BACK_DIST = 0.02
    PUSH_FORWARD_DIST = 0.07

    GRASP_SUCCESS_REWARD = 1
    GRASP_FAIL_REWARD = -0.3
    PUSH_SUCCESS_REWARD = 0.5
    PUSH_FAIL_REWARD = -0.3
    DEPTH_CHANGE_THRESHOLD = 0.01
    DEPTH_CHANGE_COUNTER_THRESHOLD = 1000

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, models: Models, camera: Camera, vis=False, num_objs=3, gripper_type='85') -> None:
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.num_objs = num_objs
        self.camera = camera

        if gripper_type not in ('85', '140'):
            raise NotImplementedError('Gripper %s not implemented.' % gripper_type)
        self.gripper_type = gripper_type

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        # define the robot
        # see https://github.com/
        # bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.pandaEndEffectorIndex = 11  # 8
        self.pandaNumDofs = 7
        self.ll = [-7] * self.pandaNumDofs
        # upper limits for null space (todo: set them to proper range)
        self.ul = [7] * self.pandaNumDofs
        # joint ranges for null space (todo: set them to proper range)
        self.jr = [7] * self.pandaNumDofs
        # restposes for null space
        self.rp = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

        self.panda = p.loadURDF("./urdf/panda.urdf", (0, 0.5, 0),
                                p.getQuaternionFromEuler((0, 0, math.pi)),
                                useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.panda,
                               9,
                               self.panda,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.reset_robot()  # Then, move back

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)  # -1.57 yaw
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, 0)  # -3.14 pitch
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.04, 0.04)

        self.boxID = p.loadURDF("./urdf/skew-box-button.urdf",
                                [0.0, 0.0, 0.0],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                "controllable"])
        joints = AttrDict()
        for i in range(p.getNumJoints(self.panda)):
            info = p.getJointInfo(self.panda, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = None
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = False
            info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            print(p.getDynamicsInfo(self.panda, i))
            print(info)

        for i in range(p.getNumJoints(self.boxID)):
            info = p.getJointInfo(self.boxID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = None
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = False
            info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            print(p.getDynamicsInfo(self.boxID, i))
            print(info)
        p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=10)
        p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, position: tuple, angle: float, action_type: str):
        """
        position [x y z]: The axis in real-world coordinate
        angle: float,   for grasp, it should be in [-pi/2, pi/2)
                        for push,  it should be in [0, 2pi)
        """
        x, y, z, roll, pitch, yaw, gripper_opening_length = self.read_debug_parameter()
        pos = (x, y, z)
        orn = p.getQuaternionFromEuler((roll, pitch, yaw))
        jointPoses = p.calculateInverseKinematics(self.panda, self.pandaEndEffectorIndex, pos, orn, self.ll, self.ul,
                                                  self.jr, self.rp, maxNumIterations=20)
        # arm
        for i in range(self.pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
        # fingers, [0, 0.04]
        for i in [9, 10]:
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, gripper_opening_length, force=100)
        # rgb, depth, seg = self.camera.shot()
        # observation = (rgb, depth, seg)
        observation = None
        reward = -100
        done = False
        info = {}
        return observation, reward, done, info

    def reset_robot(self):

        index = 0
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC:
                p.resetJointState(self.panda, j, self.rp[index])
                index = index + 1
            if jointType == p.JOINT_REVOLUTE:
                p.resetJointState(self.panda, j, self.rp[index])
                index = index + 1

        for _ in range(10):
            self.step_simulation()

    def reset_box(self):
        pass

    def reset(self):
        self.reset_robot()
        self.reset_box()
        rgb, depth, seg = self.camera.shot()
        return rgb, depth, seg

    def close(self):
        p.disconnect(self.physicsClient)
