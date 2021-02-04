import os
import csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gym
import numpy as np
import tensorflow as tf

from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler, UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines.common.vec_env import DummyVecEnv


def load_insts(path):
    feats = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            f = []
            for i in range(len(row)):
                if i != 0:
                    try:
                        f.append(float(row[i]))
                    except:
                        f.append(row[i])
            feats.append(f)
    return feats


class PointMassExperiment(AbstractExperiment):
    LOWER_CONTEXT_BOUNDS = np.array([-4.0, 0.5, 0.0])
    UPPER_CONTEXT_BOUNDS = np.array([4.0, 8.0, 4.0])

    INITIAL_MEAN = np.array([0.0, 4.25, 2.0])
    INITIAL_VARIANCE = np.diag(np.square([2, 1.875, 1]))

    TARGET_MEAN = np.array([2.5, 0.5, 0.0])
    TARGET_VARIANCE = np.diag(np.square([4e-3, 3.75e-3, 2e-3]))

    DISCOUNT_FACTOR = 0.95
    STD_LOWER_BOUND = np.array([0.2, 0.1875, 0.1])
    KL_THRESHOLD = 8000.0
    MAX_KL = 0.05

    ZETA = {Learner.TRPO: 1.6, Learner.PPO: 1.6, Learner.SAC: 1.8}
    ALPHA_OFFSET = {Learner.TRPO: 70, Learner.PPO: 70, Learner.SAC: 50}
    OFFSET = {Learner.TRPO: 5, Learner.PPO: 5, Learner.SAC: 5}

    STEPS_PER_ITER = 2048
    LAM = 0.99

    AG_P_RAND = {Learner.TRPO: 0.1, Learner.PPO: 0.1, Learner.SAC: 0.1}
    AG_FIT_RATE = {Learner.TRPO: 100, Learner.PPO: 100, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.TRPO: 1000, Learner.PPO: 500, Learner.SAC: 1000}

    GG_NOISE_LEVEL = {Learner.TRPO: 0.05, Learner.PPO: 0.025, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.TRPO: 200, Learner.PPO: 200, Learner.SAC: 100}
    GG_P_OLD = {Learner.TRPO: 0.2, Learner.PPO: 0.1, Learner.SAC: 0.1}

    def __init__(
        self,
        base_log_dir,
        curriculum_name,
        learner_name,
        parameters,
        seed,
        instance_file=None,
        test_file=None,
    ):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        if instance_file and not test_file:
            raise RuntimeError("Instance-based evaluation needs both eval and test set")
        elif instance_file and test_file:
            self.eval_env, self.vec_eval_env = self.create_environment(
                evaluation=True, file=instance_file
            )
            self.test_env, self.vec_test_env = self.create_environment(
                test=True, file=test_file
            )
        else:
            self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False, test=None, file=None):
        env = gym.make("ContextualPointMass-v1")
        if (evaluation or self.curriculum.default()) and not file:
            teacher = GaussianSampler(
                self.TARGET_MEAN.copy(),
                self.TARGET_VARIANCE,
                (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()),
            )
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif evaluation and file:
            instances = load_insts(file)
            env = BaseWrapper(
                env,
                None,
                self.DISCOUNT_FACTOR,
                context_visible=True,
                mode=1,
                instance_features=instances,
            )
        elif test and file:
            instances = load_insts(file)
            env = BaseWrapper(
                env,
                None,
                self.DISCOUNT_FACTOR,
                context_visible=True,
                mode=1,
                instance_features=instances,
            )
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(
                self.LOWER_CONTEXT_BOUNDS.copy(),
                self.UPPER_CONTEXT_BOUNDS.copy(),
                seed=self.seed,
                fit_rate=self.AG_FIT_RATE[self.learner],
                random_task_ratio=self.AG_P_RAND[self.learner],
                max_size=self.AG_MAX_SIZE[self.learner],
            )
            env = ALPGMMWrapper(
                env, teacher, self.DISCOUNT_FACTOR, context_visible=True
            )
        elif self.curriculum.goal_gan():
            samples = np.clip(
                np.random.multivariate_normal(
                    self.INITIAL_MEAN, self.INITIAL_VARIANCE, size=1000
                ),
                self.LOWER_CONTEXT_BOUNDS,
                self.UPPER_CONTEXT_BOUNDS,
            )
            teacher = GoalGAN(
                self.LOWER_CONTEXT_BOUNDS.copy(),
                self.UPPER_CONTEXT_BOUNDS.copy(),
                state_noise_level=self.GG_NOISE_LEVEL[self.learner],
                success_distance_threshold=0.01,
                update_size=self.GG_FIT_RATE[self.learner],
                n_rollouts=2,
                goid_lb=0.25,
                goid_ub=0.75,
                p_old=self.GG_P_OLD[self.learner],
                pretrain_samples=samples,
            )
            env = GoalGANWrapper(
                env, teacher, self.DISCOUNT_FACTOR, context_visible=True
            )
        elif self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(
                self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner]
            )
            bounds = (
                self.LOWER_CONTEXT_BOUNDS.copy(),
                self.UPPER_CONTEXT_BOUNDS.copy(),
            )
            teacher = SelfPacedTeacher(
                self.TARGET_MEAN.copy(),
                self.TARGET_VARIANCE.copy(),
                self.INITIAL_MEAN.copy(),
                self.INITIAL_VARIANCE.copy(),
                bounds,
                alpha_fn,
                max_kl=self.MAX_KL,
                std_lower_bound=self.STD_LOWER_BOUND.copy(),
                kl_threshold=self.KL_THRESHOLD,
                use_avg_performance=True,
            )
            env = SelfPacedWrapper(
                env, teacher, self.DISCOUNT_FACTOR, context_visible=True
            )
        elif self.curriculum.random():
            teacher = UniformSampler(
                self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()
            )
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(
            common=dict(
                gamma=self.DISCOUNT_FACTOR,
                n_cpu_tf_sess=1,
                seed=self.seed,
                verbose=0,
                policy_kwargs=dict(layers=[21], act_fun=tf.tanh),
            ),
            trpo=dict(
                max_kl=0.004,
                timesteps_per_batch=self.STEPS_PER_ITER,
                lam=self.LAM,
                vf_stepsize=0.23921693516009684,
            ),
            ppo=dict(
                n_steps=self.STEPS_PER_ITER,
                noptepochs=4,
                nminibatches=4,
                lam=self.LAM,
                max_grad_norm=None,
                vf_coef=1.0,
                cliprange_vf=-1,
                ent_coef=0.0,
            ),
            sac=dict(
                learning_rate=3e-4,
                buffer_size=10000,
                learning_starts=500,
                batch_size=64,
                train_freq=1,
                target_entropy="auto",
            ),
        )

    def create_experiment(self):
        timesteps = 1000 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        vec_env = Monitor(env, self.log_dir)
        model, interface = self.learner.create_learner(
            vec_env, self.create_learner_params()
        )

        if isinstance(env.teacher, SelfPacedTeacher):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {
            "learner": interface,
            "env_wrapper": env,
            "sp_teacher": sp_teacher,
            "n_inner_steps": 1,
            "n_offset": self.OFFSET[self.learner],
            "save_interval": 5,
            "step_divider": self.STEPS_PER_ITER if self.learner.sac() else 1,
        }
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        alpha_fn = PercentageAlphaFunction(
            self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner]
        )
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        return SelfPacedTeacher(
            self.TARGET_MEAN.copy(),
            self.TARGET_VARIANCE.copy(),
            self.INITIAL_MEAN.copy(),
            self.INITIAL_VARIANCE.copy(),
            bounds,
            alpha_fn,
            max_kl=self.MAX_KL,
            std_lower_bound=self.STD_LOWER_BOUND,
            kl_threshold=self.KL_THRESHOLD,
            use_avg_performance=True,
        )

    def get_env_name(self):
        return "point_mass"

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)

        for _ in range(100):
            # Eval set evaluation
            self.eval_hook(model, self.vec_eval_env, 0, path, "eval")
            # Test set evaluation
            self.eval_hook(model, self.vec_test_env, 0, path, "test")

        return self.eval_env.get_statistics()[1], self.test_env.get_statistics()[1]

    def eval_hook(self, model, eval_env, s, outdir, name):
        step = 1
        to_eval = eval_env.env_method("get_num_insts")[0]
        train_reward = 0
        policies = []
        rewards = []
        for i in range(to_eval):
            obs = eval_env.reset()
            done = False
            rews = 0
            pol = [i]
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, r, done, infos = self.vec_eval_env.step(action)
                pol.append(action)
                rews += r
            rewards.append(rews)
            train_reward += rews
            policies.append(pol)
        train_reward = train_reward / to_eval
        with open(os.path.join(outdir, f"{name}_reward.txt"), "a") as fh:
            fh.writelines(
                str(train_reward)
                + "\t"
                + str(step)
                + "\t"
                + str(s)
                + "\t"
                + str(to_eval)
                + "\n"
            )

        with open(os.path.join(outdir, f"{name}_per_instance.txt"), "a") as fh:
            r_string = ""
            for r in rewards:
                r_string += str(r) + "\t"
            r_string += str(step) + "\t" + str(s) + "\n"
            fh.writelines(r_string)

        with open(os.path.join(outdir, f"{name}_policies.txt"), "a") as fh:
            for p in policies:
                p_string = str(p[0]) + " "
                for i in range(1, len(p)):
                    p_string += str(p[i])
                fh.writelines(p_string + "\n")
