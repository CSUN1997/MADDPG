import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Tanh()
        )

    def forward(self, state):
        output = self.model(state)
        return output


# class Critic(nn.Module):
#     def __init__(self, state_space, action_space):
#         super(Critic, self).__init__()
#         self.state = nn.Linear(state_space, 256)
#         self.model = nn.Sequential(
#             nn.Linear(256 + action_space, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, state, action):
#         output = self.state(state)
#         output = self.model(torch.cat([output, action], dim=1))
#         return output


class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space + action_space, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        # output = self.state(state)
        output = self.model(torch.cat([state, action], dim=1))
        return output


class MADDPG:
    def __init__(self, state_size, action_size, n_agent, gamma=0.99,
                 lr_actor=0.01, lr_critic=0.05, update_freq=200):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agent = n_agent
        self.gamma = gamma
        self.update_freq = update_freq
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device {}".format(self.device))

        self.actors = [Actor(state_size, action_size).to(self.device) for _ in range(n_agent)]
        self.actors_target = [Actor(state_size, action_size).to(self.device) for _ in range(n_agent)]

        self.critics = [Critic(state_size * n_agent, action_size * n_agent).to(self.device) for _ in range(n_agent)]
        self.critics_target = [Critic(state_size * n_agent, action_size * n_agent).to(self.device) for _ in range(n_agent)]
        # self.critic = Critic(state_size * n_agent, action_size * n_agent).to(self.device)
        # self.critic_target = Critic(state_size * n_agent, action_size * n_agent).to(self.device)

        [actor_target.eval() for actor_target in self.actors_target]
        [critic_target.eval() for critic_target in self.critics_target]

        self.actors_optim = [optim.Adam(actor.parameters(), lr_actor) for actor in self.actors]
        self.critics_optim = [optim.Adam(critic.parameters(), lr_critic) for critic in self.critics]
        # self.actor_optim = optim.Adam(sum([list(actor.parameters()) for actor in self.actors]))
        # self.critic_optim = optim.Adam(sum([list(critic.parameters()) for critic in self.critics]))

        self.steps = 0

    def update_target(self):
        for i in range(self.n_agent):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs
        return torch.FloatTensor(inputs).to(self.device)

    def choose_action(self, states):
        actions = [actor(self.to_tensor(state)).detach().cpu().numpy() for actor, state in zip(self.actors, states)]
        return actions

    def learn(self, s, a, r, sn, d):
        states = [self.to_tensor(state) for state in s]
        actions = [self.to_tensor(action) for action in a]
        rewards = [self.to_tensor(reward) for reward in r]
        rewards = [(reward - torch.mean(reward)) / torch.std(reward) for reward in rewards]
        states_next = [self.to_tensor(state_next) for state_next in sn]
        dones = [self.to_tensor(done.astype(int)) for done in d]
        all_state = torch.cat(states, dim=1)
        all_action = torch.cat(actions, dim=1)
        all_state_next = torch.cat(states_next, dim=1)
        actor_losses = 0
        for i in range(self.n_agent):
            cur_action = all_action.clone()
            action = self.actors[i](states[i])
            action_size = action.shape[1]
            cur_action[:, action_size * i: action_size * (i + 1)] = action
            actor_loss = -torch.mean(self.critics[i](all_state, cur_action))
            actor_losses += actor_loss

        actions_next = [actor_target(state_next).detach() for state_next, actor_target in zip(states_next, self.actors_target)]
        all_action_next = torch.cat(actions_next, dim=1)
        critic_losses = 0
        for i in range(self.n_agent):
            next_value = self.critics_target[i](all_state_next, all_action_next)
            Q = self.critics[i](all_state, all_action)
            target = rewards[i] + self.gamma * next_value.detach()
            critic_loss = F.mse_loss(Q, target)
            critic_losses += critic_loss

        # actor
        # self.actor_optim.zero_grad()
        [actor_optim.zero_grad() for actor_optim in self.actors_optim]
        actor_losses.backward()
        [nn.utils.clip_grad_norm_(actor.parameters(), 0.3) for actor in self.actors]
        # self.actor_optim.step()
        [actor_optim.step() for actor_optim in self.actors_optim]
        # critic
        # self.critic_optim.zero_grad()
        [critic_optim.zero_grad() for critic_optim in self.critics_optim]
        critic_losses.backward()
        [nn.utils.clip_grad_norm_(critic.parameters(), 0.3) for critic in self.critics]
        # self.critic_optim.step()
        [critic_optim.step() for critic_optim in self.critics_optim]

        # update target networks
        if self.steps % self.update_freq == 0:
            self.update_target()
        self.steps += 1
        # print(actor_losses, critic_losses)

        return (actor_losses + critic_losses).item()

    # def save_model(self, directory):
    #     save_content = {}
    #     for i in range(self.n_agent):
    #         save_content['actor_{}'.format(i)] = self.actors[i].state_dict()
    #         save_content['critic_{}'.format(i)] = self.critics[i].state_dict()
    #         save_content['actor_optimizer_{}'.format(i)] = self.actors_optim[i].state_dict()
    #         save_content['critic_optimizer_{}'.format(i)] = self.critics_optim[i].state_dict()
    #     torch.save(save_content, directory + "MAAC_model")
    #
    # def load_model(self, directory):
    #     saved_content = torch.load(directory + "MAAC_model")
    #     for i in range(self.n_agent):
    #         self.actors[i].load_state_dict(saved_content['actor_{}'.format(i)])
    #         self.critics[i].load_state_dict(saved_content['critic_{}'.format(i)])
    #         self.actors_optim[i].load_state_dict(saved_content['actor_optimizer_{}'.format(i)])
    #         self.critics_optim[i].load_state_dict(saved_content['critic_optimizer_{}'.format(i)])
    #     self.update_target()