from numpy import multiply
import pybullet as p
import math
from collections import namedtuple
import time


class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controlable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controlable joints on the arm

            gripper_range: List[Min, Max]


        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.arm_controlable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.arm_controlable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.arm_controlable_joints) >= self.arm_num_dofs
        self.arm_controlable_joints = self.arm_controlable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controlable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controlable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.arm_controlable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)


class Panda(RobotBase):
    def __init_robot__(self):
        # define the robot
        # see https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.eef_id = 11
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.id = p.loadURDF("./urdf/panda.urdf", self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.04]
        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.id,
                               9,
                               self.id,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_gripper(self, open_length):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in [9, 10]:
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, open_length, force=20)


class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF("./urdf/ur5_robotiq_85.urdf", self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        # move child joints
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, targetPosition=open_angle * multiplier,
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)