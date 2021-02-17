from random import random, seed
import time
import pybullet as p
import pybullet_data
from datetime import datetime
import numpy as np
from numpy import float32, inf
from gym.spaces import Box

seed(datetime.now())

TARGET_LOC          = np.array([0.0, 0.0, 0.18])
TARGET_ORIENT       = np.array([1,1,0])
JOINT_AT_LIMIT_COST = 0.1
TORQUE_COST         = 0.4
STEP_ACTION_RATE    = 5
REWARD_SCALE        = 10
GROUND_CONTACT_COST = 100

class Env:
    def __init__(
            self,
            name,
            var=0.1,
            vis=False
        ):
        self.var = var
        self.vis = vis
        self.name = name
        self.last_state = None
        self.current_state = None
        self.client = p.connect(p.GUI) if vis else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()
        self.describe_space()

    def describe_space(self):
        all_state = self._get_state()
        joint_lower_bounds, joint_upper_bounds = [], []
        num_joints = p.getNumJoints(self.robot_id)

        for joint_i in range(num_joints):
            lower, upper = p.getJointInfo(self.robot_id, joint_i)[8:10]
            joint_lower_bounds.append(lower)
            joint_upper_bounds.append(upper)

        obs_space_upper_bounds = joint_upper_bounds \
            + [inf for _ in range(num_joints, len(all_state))]
        obs_space_lower_bounds = joint_lower_bounds \
            + [inf for _ in range(num_joints, len(all_state))]
        self.observation_space = Box((len(obs_space_upper_bounds),),
                                     obs_space_upper_bounds,
                                     obs_space_lower_bounds)

        self.action_space = Box((num_joints, ),
                                np.array(joint_upper_bounds, dtype=float32),
                                np.array(joint_lower_bounds, dtype=float32))

    def reset(self):
        self.plane_id = None
        self.robot_id = None
        for body_id in range(p.getNumBodies()):
            p.removeBody(body_id)

        slope = p.getQuaternionFromEuler([
            self.var*random() - self.var/2,
            self.var*random() - self.var/2,
            0])

        # slope = p.getQuaternionFromEuler([0, 0, 0])

        self.plane_id = p.loadURDF(
            "plane.urdf",
            [0, 0, 0],
            slope)

        robot_start_pos = [0, 0, 0.25]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "gym/urdf/robot-simple.urdf",
            robot_start_pos,
            robot_start_orientation)

        p.setGravity(0, 0, -10)
        state = self._get_state()
        self.last_state = state
        self.current_state = state
        return state

    def take_action(self, actions):
        for joint_i, action in enumerate(actions):
            maxForce = 175
            p.setJointMotorControl2(self.robot_id, joint_i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=action,
                                    force=maxForce)

    def step(self, actions):
        self.last_state = self.current_state
        for _ in range(STEP_ACTION_RATE):
            self.take_action(actions)
            p.stepSimulation()
        # note _get_state must happen before _get_reward or _get_reward
        # will return nonsense!
        self.current_state = self._get_state()
        reward, done = self._get_reward()
        return self.current_state, reward, done, None

    def _get_state(self):
        state_ls = [p.getLinkState(self.robot_id, i)[0]
                    for i in range(p.getNumJoints(self.robot_id))]
        base_link_state = p.getBasePositionAndOrientation(self.robot_id)[0]
        state = np.array([
            *[p.getJointState(self.robot_id, i)[0]
              for i in range(p.getNumJoints(self.robot_id))],
            *[item for subls in state_ls for item in subls],
            *base_link_state
        ])
        return state

    def _standing_reward(self):
        """Dependent on base link difference from target location and
        target orientation. If base link contacts ground 100 point penalty is
        added and a small cost is added per unit of torque used.
        """

        base_data = p.getBasePositionAndOrientation(self.robot_id)
        base_loc = np.array(base_data[0])
        orient = np.array(p.getEulerFromQuaternion(base_data[1]))

        dist_from_target =  np.linalg.norm(base_loc - TARGET_LOC) \
            + 6 * np.linalg.norm(orient * TARGET_ORIENT)
        return REWARD_SCALE/max(dist_from_target, 0.01)

    def _torque_cost(self):
        torque_sum = sum([abs(p.getJointState(self.robot_id, i)[3])/1500
                          for i in range(p.getNumJoints(self.robot_id))])
        return - torque_sum * TORQUE_COST

    def _check_done(self):
        if p.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=-1,
                bodyB=self.plane_id,
                linkIndexB=-1):
            return True
        return False

    def _get_reward(self):
        costs = np.array([
            self._joints_at_limit_cost(),
            # self._standing_reward(),
            self._progress_reward(),
            self._torque_cost()
        ])
        done = self._check_done()
        return (costs.sum(), done) if not done \
            else (costs.sum() - GROUND_CONTACT_COST, done)

    def _progress_reward(self):
        forwards_movement = self.current_state[-2] - self.last_state[-2]
        return forwards_movement * REWARD_SCALE

    def close(self):
        p.disconnect()

    def _joints_at_limit_cost(self):
        num_joints = p.getNumJoints(self.robot_id)
        count = 0
        for joint_i, j_rad in enumerate(self.current_state[:num_joints]):
            joint_per_loc = \
                (j_rad + abs(self.observation_space.low[joint_i]))  / \
                self.observation_space.arc_sizes[joint_i]
            if joint_per_loc < 0.05 or joint_per_loc > 0.95:
                count += 1
        return - count * JOINT_AT_LIMIT_COST
