import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, setup_sisbot, setup_sisbot_force, Camera


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
        self.robotID = p.loadURDF("./urdf/ur5_robotiq_%s.urdf" % gripper_type,
                                  [0, 0.5, -0.8],  # StartPosition
                                  p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
                                  useFixedBase=True,
                                  flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
            setup_sisbot(p, self.robotID, gripper_type)
        self.eefID = 7  # ee_link
        # Add force sensors
        p.enableJointForceTorqueSensor(self.robotID, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(self.robotID, self.joints['right_inner_finger_pad_joint'].id)
        # Change the friction of the gripper
        # p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=100, spinningFriction=10, rollingFriction=10)
        # p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=100, spinningFriction=10, rollingFriction=10)
        p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=0.5)
        p.changeDynamics(self.robotID, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=0.5)
        # for (k, v) in self.joints.items():
        #     print(k, v)

        # Do NOT reset robot before loading objects
        self.obj_ids = []
        self.successful_obj_ids = []
        self.obj_state = []
        self.models = models
        self.models.load_objects()
        self.load_objects(self.num_objs)
        self.save_obj_state()
        self.reset_robot()  # Then, move back

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # Task space (Cartesian space)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)  # -1.57 yaw
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, 0)  # -3.14 pitch
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.085)

        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.085)
        self.ee_position_limit = ((-0.224, 0.224),
                                  (-0.224, 0.224),
                                  (0, 1))
        # Observation buffer
        self.prev_observation = tuple()

        self.boxID = p.loadURDF("./urdf/skew-box-button.urdf",
                                [0.0, 0.0, 0.0],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)


    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def load_objects(self, num):
        for _ in range(num):
            vis_shape, col_shape = random.choice(self.models)
            obj_handle = p.createMultiBody(baseMass=0.2,
                                           baseInertialFramePosition=[0, 0, 0],
                                           baseCollisionShapeIndex=col_shape,
                                           baseVisualShapeIndex=vis_shape,
                                           # Leave 0.05 space for safety
                                           basePosition=(np.clip(np.random.normal(0, 0.005), -0.2, 0.2),
                                                         np.clip(np.random.normal(0, 0.005) - 0.5, -0.7, -0.3),
                                                         self.OBJECT_INIT_HEIGHT),
                                           # useMaximalCoordinates=True,
                                           baseOrientation=p.getQuaternionFromEuler((np.random.uniform(-np.pi, np.pi),
                                                                                     np.random.uniform(0, np.pi),
                                                                                     np.random.uniform(-np.pi, np.pi)))
                                           )
            p.changeDynamics(obj_handle, -1, lateralFriction=1, rollingFriction=0.01, spinningFriction=0.001,
                             restitution=0.01)
            self.obj_ids.append(obj_handle)
            # To make the objects get a initial speed
            for _ in range(10):
                self.step_simulation()
            self.wait_until_still()
        assert len(self.obj_ids) == self.num_objs
        for obj_handle in self.obj_ids:
            p.changeDynamics(obj_handle, -1, lateralFriction=1, rollingFriction=0.01, spinningFriction=0.001,
                             restitution=0)

    @staticmethod
    def is_still(handle):
        still_eps = 1e-3
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        # print(np.abs(lin_vel).sum() + np.abs(ang_vel).sum())
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def wait_until_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if np.all(list(self.is_still(handle) for handle in self.obj_ids)):
                return
        print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' % max_wait_epochs)

    def save_obj_state(self):
        if len(self.obj_state) > 0:
            assert self.num_objs == len(self.obj_state)
            print('Warning: There is previous state available. Overwriting...')
            print(self.obj_state)
            self.obj_state = []
        assert len(self.obj_ids) == self.num_objs
        for obj_handle in self.obj_ids:
            pos, orn = p.getBasePositionAndOrientation(obj_handle)
            lin_vel, ang_vel = p.getBaseVelocity(obj_handle)
            self.obj_state.append((pos, orn, lin_vel, ang_vel))

    def reset_obj_state(self):
        assert self.num_objs == len(self.obj_state)
        for obj_handle, (pos, orn, lin_vel, ang_vel) in zip(self.obj_ids, self.obj_state):
            p.resetBasePositionAndOrientation(obj_handle, pos, orn)
            p.resetBaseVelocity(obj_handle, lin_vel, ang_vel)
        for _ in range(100):
            self.step_simulation()

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

    def check_depth_change(self, cur_depth):
        _, prev_depth, _ = self.prev_observation
        changed_depth = cur_depth - prev_depth
        changed_depth_counter = np.sum(np.abs(changed_depth) > self.DEPTH_CHANGE_THRESHOLD)
        print('changed depth pixel count:', changed_depth_counter)
        return changed_depth_counter > self.DEPTH_CHANGE_COUNTER_THRESHOLD

    def step(self, position: tuple, angle: float, action_type: str, debug: bool = False):
        """
        position [x y z]: The axis in real-world coordinate
        angle: float,   for grasp, it should be in [-pi/2, pi/2)
                        for push,  it should be in [0, 2pi)
        """
        x, y, z, roll, pitch, yaw, gripper_opening_length = self.read_debug_parameter()
        roll, pitch = 0, np.pi / 2
        if not debug:
            x, y, z = position
            yaw = angle if action_type == 'grasp' else 0.0
        orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        if debug:
            self.move_ee((x, y, z, orn))
            # self.move_gripper(gripper_opening_length, 1)
            return
        # The return value of the step() method
        observation, reward, done, info = None, 0.0, False, dict()
        self.reset_robot()
        self.move_ee((x, y, self.GRIPPER_MOVING_HEIGHT, orn))  # Top-Down grasp / push
        grasp_success, push_success = False, False
        if action_type == 'grasp':
            self.open_gripper()
            self.move_ee((x, y, z + self.GRASP_POINT_OFFSET_Z, orn),
                         custom_velocity=0.05, max_step=1000)
            # item_in_gripper = self.close_gripper(check_contact=True)
            item_in_gripper = self.close_gripper(check_contact=True)
            print('Item in Gripper!')
            # When lifting the object, constantly try to close the gripper, in case of dropping
            self.move_ee((x, y, z + self.GRASP_POINT_OFFSET_Z + 0.1, orn), try_close_gripper=False,
                         custom_velocity=0.05, max_step=1000)
            # Lift 10 cm
            if item_in_gripper:
                grasped_ids = self.check_grasped_id()
                for item_id in grasped_ids:
                    self.successful_obj_ids.append(item_id)
                    print('Successful item ID:', item_id)
                    reward += self.GRASP_SUCCESS_REWARD
                    grasp_success = True

            self.move_ee((x, y, self.GRIPPER_GRASPED_LIFT_HEIGHT, orn), try_close_gripper=False, max_step=1000)

        elif action_type == 'push':
            push_start_xy = (x - np.sin(angle) * self.PUSH_BACK_DIST,
                             y + np.cos(angle) * self.PUSH_BACK_DIST)
            push_end_xy = (x + np.sin(angle) * self.PUSH_FORWARD_DIST,
                           y - np.cos(angle) * self.PUSH_FORWARD_DIST)

            self.close_gripper()
            self.move_ee((*push_start_xy, self.GRIPPER_MOVING_HEIGHT, orn))
            _, (real_xyz, real_xyzw) = self.move_ee((*push_start_xy, 0.83, orn),
                                                    check_collision_config=dict(bool_operator='or', force=100),
                                                    custom_velocity=0.2, verbose=False, max_step=1000)
            # Move linearly towards the target - IK will cause problem!
            # for step_idx in range(1, 120):
            #     push_step_xy = (real_xyz[0] + (push_end_xy[0] - real_xyz[0]) / step_idx * 120,
            #                     real_xyz[1] + (push_end_xy[1] - real_xyz[1]) / step_idx * 120)
            #     self.move_ee((*push_step_xy, real_xyz[2], orn), custom_velocity=0.2, verbose=True)
            # Move towards the target
            self.move_ee((*push_end_xy, real_xyz[2], orn), custom_velocity=0.2, verbose=False)
            self.move_ee((*push_end_xy, self.GRIPPER_MOVING_HEIGHT, orn))

        self.move_away_arm()
        self.open_gripper()
        # TODO: Check if object is outside the environment
        rgb, depth, seg = self.camera.shot()
        depth_changed = self.check_depth_change(depth)
        if action_type == 'grasp' and not grasp_success:
            reward += self.GRASP_FAIL_REWARD
        if action_type == 'push':
            if depth_changed:
                reward += self.PUSH_SUCCESS_REWARD
            else:
                reward += self.PUSH_FAIL_REWARD
        observation = (rgb, depth, seg)
        self.prev_observation = observation
        done = (len(self.successful_obj_ids) == len(self.obj_ids))
        return observation, reward, done, info

    def reset_robot(self):
        user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        for _ in range(100):
            for i, name in enumerate(self.controlJoints):
                if i == 6:
                    self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                    break
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                self.step_simulation()

    def reset(self):
        self.reset_robot()
        self.reset_obj_state()
        self.move_away_arm()
        rgb, depth, seg = self.camera.shot()
        self.prev_observation = (rgb, depth, seg)
        self.reset_robot()
        self.successful_obj_ids = []
        return rgb, depth, seg

    def move_away_arm(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                    targetPosition=0., force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
            self.step_simulation()

    def check_grasped_id(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robotID, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robotID, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left + contact_right if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            print('Warning: Multiple items in hand!')
        if len(contact_ids) == 0:
            print(contact_left, contact_right)
        return list(item_id for item_id in contact_ids if item_id in self.obj_ids)

    def gripper_contact(self, bool_operator='and', force=100):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robotID, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robotID, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robotID, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robotID, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
            self.step_simulation()

    def open_gripper(self, step: int = 120):
        self.move_gripper(0.085, step)

    def close_gripper(self, step: int = 120, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robotID, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self.gripper_contact():
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.move_gripper(current_target_open_length - 0.005)
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.controlGripper(stop=True)
                return True
        return False

    def move_ee(self, action, max_step=500, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix
        for _ in range(max_step):
            # apply IK
            joint_poses = p.calculateInverseKinematics(self.robotID, self.eefID, [x, y, z], orn,
                                                       maxNumIterations=100, jointDamping=jd
                                                       )
            for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            self.step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)
            # Check if contact with objects
            if check_collision_config and self.gripper_contact(**check_collision_config):
                print('Collision detected!', self.check_grasped_id())
                return False, p.getLinkState(self.robotID, self.eefID)[0:2]
            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(self.robotID, self.eefID)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        print('Failed to reach the target')
        return False, p.getLinkState(self.robotID, self.eefID)[0:2]

    def close(self):
        p.disconnect(self.physicsClient)
