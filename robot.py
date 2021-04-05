import pybullet as p
import math


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

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controlable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controlable joints on the arm


        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.__init_robot__()

        # Get controlable joints
        self.arm_controlable_joints = []
        for j in range(p.getNumJoints(self.id)):
            # p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.id, j)
            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.arm_controlable_joints.append(j)
        assert len(self.arm_controlable_joints) >= self.arm_num_dofs
        self.arm_controlable_joints = self.arm_controlable_joints[:self.arm_num_dofs]

        assert self.arm_num_dofs == len(self.arm_lower_limits) == len(self.arm_upper_limits) \
            == len(self.arm_joint_ranges) == len(self.arm_rest_poses)

    def __init_robot__(self):
        raise NotImplementedError

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

    def reset_gripper():
        pass

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn, self.ll,
                                                       self.ul, self.jr, self.rp, maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i in range(self.arm_num_dofs):
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, joint_poses[i], force=5 * 240.)

        # TODO: USE setJointMotorControlArray
        # TODO: Add max_forces, max_vels, etc...



class Panda(RobotBase):
    def __init_robot__(self):
        # define the robot
        # see https://github.com/
        # bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.eef_id = 11
        self.arm_num_dofs = 7
        self.arm_lower_limits = [-7] * self.arm_num_dofs
        # upper limits for null space (todo: set them to proper range)
        self.arm_upper_limits = [7] * self.arm_num_dofs
        # joint ranges for null space (todo: set them to proper range)
        self.arm_joint_ranges = [7] * self.arm_num_dofs
        # restposes for null space
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.id = p.loadURDF("./urdf/panda.urdf", self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
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

