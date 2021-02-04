import numpy as np
from gym import spaces
import sys
from pybulletgym.envs.mujoco.envs.locomotion.ant_env import (
    AntMuJoCoEnv as AntEnv,
)


class AntGoalWrapper(AntEnv):
    def __init__(
        self, instance_feats, test, action_space_size=10, external_eval=False
    ):
        self.instances = instance_feats
        self.inst_id = -1
        if test:
            self.instance_set_size = 1
            self.curr_set = instance_feats
            self.indices = np.arange(len(instance_feats))
        else:
            self.instance_set_size = 1 / len(self.instances)
            self.curr_set = [self.instances[0]]
            self.indices = [0]
        self.env = AntEnv()
        self.robot = self.env.robot
        self.action_space = self.env.action_space
        obs_size = 3 + len(self.env.observation_space.low)
        self.observation_space = spaces.Box(
            -np.inf * np.ones(obs_size), np.inf * np.ones(obs_size)
        )
        self.test = test
        self.eval = external_eval
        self.c_step = 0
        self.x_goal = None
        self.y_goal = None
        self.ignore_joint = None
        self.sampler = None
        self.ownsPhysicsClient = False

    def reset(self):
        self.c_step = 0
        self.inst_id = (self.inst_id + 1) % len(self.curr_set)
        self.x_goal = self.curr_set[self.inst_id][0]
        self.y_goal = self.curr_set[self.inst_id][1]
        self.ignore_joint = int(self.curr_set[self.inst_id][2])
        #if self.sampler is not None:
        #    self.x_goal, self.y_goal = self.sampler()
        self.env = AntEnv()
        self.env.robot.walk_target_x = self.x_goal
        self.env.robot.walk_target_y = self.y_goal
        obs = self.env.reset()
        next_state = np.concatenate(([self.x_goal, self.y_goal, self.ignore_joint], obs))
        return next_state

    def step(self, action):
        self.c_step += 1
        if self.eval:
            action = action[0]
        if self.ignore_joint > 0:
            action[self.ignore_joint*2] = 0
            action[self.ignore_joint*2+1] = 0
        obs, r, done, _ = self.env.step(np.array(action))
        next_state = np.concatenate(([self.x_goal, self.y_goal, self.ignore_joint], obs))
        if self.c_step >= 150:
            done = True
        return next_state, r, done, {}

    def get_id(self):
        return self.inst_id

    def set_test(self, test):
        self.test = test

    def set_eval(self, ext_eval):
        self.eval = ext_eval

    def get_feats(self):
        return len(self.instances)

    def get_instances(self):
        return self.instances

    def get_instance_set(self):
        return self.indices, self.curr_set

    def get_instance_size(self):
        return min(int(np.ceil(len(self.instances) * self.instance_set_size)), len(self.instances))

    def increase_set_size(self, kappa, multi):
        if multi:
            self.instance_set_size = self.instance_set_size * kappa
        else:
            self.instance_set_size += kappa / len(self.instances)

    def set_instance_set(self, indices):
        size = int(np.ceil(len(self.instances) * self.instance_set_size))
        if size <= 0:
            size = 1
        self.curr_set = np.array(self.instances)[indices[:size]]
        self.indices = indices
