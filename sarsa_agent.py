from agent import BaseAgent

import numpy as np

class DifferentialSarsaAgent(BaseAgent):

    def __init__(self):
        self.name = "diff_sarsa_agent"  # for keeping track of different agents later

    def choose_action(self, observation):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
        Args:
            observation (List): coordinates of the agent (two elements)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[observation]
            action = self.rand_generator.choice(np.argwhere(values == np.amax(values)).flatten())

        return action

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""

        self.num_actions = agent_info.get("num_actions", 4)
        assert "num_states" in agent_info.keys()
        self.num_states = agent_info["num_states"]
        self.step_size = agent_info.get("step_size", 0.1)
        self.beta = agent_info.get("beta", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 22))

        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.avg_reward = 0.0
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1

    def agent_start(self, observation):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's env_start function.
        Returns:
            (integer) the first action the agent takes.
        """

        self.past_state = observation
        self.past_action = self.choose_action(observation)

        return self.past_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            (integer) The action the agent takes given this observation.
        """
        # Action selection
        action = self.choose_action(observation)

        # Direct RL step
        delta = reward - self.avg_reward + self.q_values[observation][action] - \
                self.q_values[self.past_state][self.past_action]
        self.q_values[self.past_state][self.past_action] += self.step_size * delta
        self.avg_reward += self.beta * delta

        self.past_state = observation
        self.past_action = action

        return self.past_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        A direct-RL update with the final transition.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        self.q_values[self.past_state][self.past_action] += self.step_size * \
                        (reward - self.avg_reward - self.q_values[self.past_state][self.past_action])

        # for i in range(6):
        #     for j in range(9):
        #         values = self.q_values[i][j]
        #         print(np.argwhere(values == np.max(values)).flatten())
        # print()
        # print(self.q_values)

class SarsaAgent(BaseAgent):

    def __init__(self):
        self.name = "sarsa_agent"  # for keeping track of different agents later

    def choose_action(self, observation):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
        Args:
            observation (List): coordinates of the agent (two elements)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[observation]
            action = self.rand_generator.choice(np.argwhere(values == np.amax(values)).flatten())

        return action

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""

        self.num_actions = agent_info.get("num_actions", 4)
        assert "num_states" in agent_info.keys()
        self.num_states = agent_info["num_states"]
        self.gamma = agent_info.get("gamma", 0.9)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 22))

        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1

    def agent_start(self, observation):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's env_start function.
        Returns:
            (integer) the first action the agent takes.
        """

        self.past_state = observation
        self.past_action = self.choose_action(observation)

        return self.past_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            (integer) The action the agent takes given this observation.
        """
        # Action selection
        action = self.choose_action(observation)

        # Direct RL step
        self.q_values[self.past_state][self.past_action] += self.step_size * \
                        (reward + self.gamma * self.q_values[observation][action] -
                         self.q_values[self.past_state][self.past_action])

        self.past_state = observation
        self.past_action = action

        return self.past_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        A direct-RL update with the final transition.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        self.q_values[self.past_state][self.past_action] += self.step_size * \
                        (reward - self.q_values[self.past_state][self.past_action])

        # for i in range(6):
        #     for j in range(9):
        #         values = self.q_values[i][j]
        #         print(np.argwhere(values == np.max(values)).flatten())
        # print()
        # print(self.q_values)