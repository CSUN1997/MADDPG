import numpy as np
import time
import random
import os
import torch

from environment.grid import LogicalEnv
from models.maac import MADDPG
# from models.maac_seperate import MADDPG
# from utils.memory import Memory
from utils.replay_buffer import Memory
from utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
import utils.general_utilities as general_utilities
from config import Config


def play(is_testing):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_theta_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_mu_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_sigma_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_dt_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_x0_{}".format(i) for i in range(env.n_agents)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    for episode in range(Config.episodes):
        states = env.reset()
        episode_losses = np.zeros(env.n_agents)
        episode_rewards = np.zeros(env.n_agents)
        collision_count = np.zeros(env.n_agents)
        steps = 0
        # print("=====================================================================")
        while True:
            steps += 1
            # act
            actions = maddpgs.choose_action(states)
            for i in range(env.n_agents):
                actions[i] = np.clip(actions[i] + actors_noise[i](), -2, 2)
                # actions.append(action)
            # print(actions)

            # step
            states_next, rewards, done = env.step(actions)
            # print([env.agents[i].pos for i in range(env.n_agents)])
            # env.visualize()
            # learn
            if not is_testing:
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < Config.batch_size else random.sample(
                    range(size), Config.batch_size)

                state_batch, action_batch, reward_batch, state_next_batch, done_batch = [], [], [], [], []
                for i in range(env.n_agents):
                    memories[i].remember(states[i], actions[i], rewards[i], states_next[i], done[i])

                    if memories[i].pointer > Config.batch_size * 10:
                        s, a, r, sn, d = memories[i].sample(batch)
                        state_batch.append(s)
                        action_batch.append(a)
                        r = np.reshape(r, (Config.batch_size, 1))
                        reward_batch.append(r)
                        state_next_batch.append(sn)
                        done_batch.append(d)

                if memories[0].pointer > Config.batch_size * 10:
                    loss = maddpgs.learn(state_batch, action_batch, reward_batch, state_next_batch, done_batch)
                    episode_losses += loss
                else:
                    episode_losses = -1 * np.ones_like(episode_losses)

            states = states_next
            episode_rewards += rewards

            # reset states if done
            if any(done):
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.extend([episode_rewards[i] for i in range(env.n_agents)])
                statistic.extend([episode_losses[i] for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].theta for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].mu for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].sigma for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].dt for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].x0 for i in range(env.n_agents)])
                statistics.add_statistics(statistic)
                if episode % 25 == 0:
                    print(statistics.summarize_last())
                break

        if episode % Config.checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(Config.csv_filename_prefix, episode))
    return statistics


if __name__ == '__main__':
    print('Start experiment for scheme {}'.format(Config.scheme))
    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)
    if not os.path.exists(Config.experiment_prefix + Config.scheme):
        os.makedirs(Config.experiment_prefix + Config.scheme)

    # for rounds in range(3):
    general_utilities.dump_dict_as_json(general_utilities.get_vars(vars(Config)),
                                        Config.experiment_prefix + "/save/run_parameters.json")

    # init env
    env = LogicalEnv(n_agents=Config.n_agents, max_step=Config.max_step)

    # Extract ou initialization values
    ou_mus = [np.zeros(env.action_space[i]) for i in range(env.n_agents)]
    ou_sigma = [0.3 for i in range(env.n_agents)]
    ou_theta = [0.15 for i in range(env.n_agents)]
    ou_dt = [1e-2 for i in range(env.n_agents)]
    ou_x0 = [None for i in range(env.n_agents)]

    # set random seed

    maddpgs = MADDPG(env.observation_space[0], env.action_space[0], env.n_agents,
                     Config.gamma, Config.lr_actor, Config.lr_critic, Config.update_freq)
    actors_noise = []
    memories = []
    for i in range(env.n_agents):
        n_action = env.action_space[i]
        state_size = env.observation_space[i]
        speed = 1

        actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=ou_mus[i],
            sigma=ou_sigma[i],
            theta=ou_theta[i],
            dt=ou_dt[i],
            x0=ou_x0[i]))
        memories.append(Memory(Config.memory_size))

    start_time = time.time()

    # play
    statistics = play(is_testing=False)
    # maddpgs.save_model("../results/model/")
    # bookkeeping
    print("Finished {} episodes in {} seconds".format(Config.episodes, time.time() - start_time))
    # tf.summary.FileWriter(args.experiment_prefix +
    #                       args.weights_filename_prefix, session.graph)
    # save_path = saver.save(session, os.path.join(
    #     args.experiment_prefix + args.weights_filename_prefix, "models"), global_step=args.episodes)
    # print("saving model to {}".format(save_path))
    statistics.dump(Config.experiment_prefix + Config.scheme + '/' + Config.csv_filename_prefix + ".csv")
