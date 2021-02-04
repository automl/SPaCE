from deep_sprl.teachers.abstract_teacher import BaseWrapper, BaseIsaacWrapper


class ALPGMMWrapper(BaseWrapper):

    def __init__(self, env, alp_gmm, discount_factor, context_visible):
        BaseWrapper.__init__(self, env, alp_gmm, discount_factor, context_visible)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        self.teacher.update(cur_context, discounted_reward)


class ALPGMMIsaacWrapper(BaseIsaacWrapper):

    def __init__(self, create_exp, alp_gmm, discount_factor, context_visible, sync_frame_time=False,
                 num_envs=40, view=False):
        super().__init__(create_exp, alp_gmm, discount_factor, context_visible, view=view,
                         sync_frame_time=sync_frame_time, num_envs=num_envs)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        self.teacher.update(cur_context, discounted_reward)
