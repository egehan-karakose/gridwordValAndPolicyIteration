import numpy as np
import matplotlib.pyplot as plt
import cv2


class GridWorldMDP:

    # up, right, down, left
    directionDeltas = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]
    numActions = len(directionDeltas)

    def __init__(self,
                 reward_grid,
                 terminal_mask,
                 obstacle_mask,
                 action_probabilities,
                 no_action_probability):

        self.reward = reward_grid
        self.terminalMask = terminal_mask
        self.obstacleMask = obstacle_mask
        self._T = self.create_transition_matrix(
            action_probabilities,
            no_action_probability,
            obstacle_mask
        )

    @property
    def shape(self):
        return self.reward.shape

    @property
    def size(self):
        return self.reward.size

    @property
    def reward_grid(self):
        return self.reward

    def generate_experience(self, current_state_idx, action_idx):
        sr, sc = self.grid_indices_to_coordinates(current_state_idx)
        next_state_probs = self._T[sr, sc, action_idx, :, :].flatten()
        np.random.seed(62)
        next_state_idx = np.random.choice(np.arange(next_state_probs.size),
                                          p=next_state_probs)

        return (next_state_idx,
                self.reward.flatten()[next_state_idx],
                self.terminalMask.flatten()[next_state_idx])

    def grid_indices_to_coordinates(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coordinates_to_indices(self, coordinates=None):
        # Annoyingly, this doesn't work for negative indices.
        # The mode='wrap' parameter only works on positive indices.
        if coordinates is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coordinates, self.shape)

    def best_policy(self, utility):
        M, N = self.shape
        best = np.argmax((utility.reshape((1, 1, 1, M, N)) * self._T)
                         .sum(axis=-1).sum(axis=-1), axis=2)

        return best

    def init_utility_policy_storage(self, depth):
        M, N = self.shape
        utilities = np.zeros((M, N, depth))
        policies = np.zeros_like(utilities)
        return utilities, policies

    def create_transition_matrix(self,
                                 action_probabilities,
                                 no_action_probability,
                                 obstacle_mask):
        M, N = self.shape

        T = np.zeros((M, N, self.numActions, M, N))

        r0, c0 = self.grid_indices_to_coordinates()

        T[r0, c0, :, r0, c0] += no_action_probability

        for action in range(self.numActions):
            for offset, P in action_probabilities:
                direction = (action + offset) % self.numActions

                dr, dc = self.directionDeltas[direction]
                r1 = np.clip(r0 + dr, 0, M - 1)
                c1 = np.clip(c0 + dc, 0, N - 1)

                temp_mask = obstacle_mask[r1, c1].flatten()
                r1[temp_mask] = r0[temp_mask]
                c1[temp_mask] = c0[temp_mask]

                T[r0, c0, action, r1, c1] += P

        terminal_locs = np.where(self.terminalMask.flatten())[0]
        T[r0[terminal_locs], c0[terminal_locs], :, :, :] = 0

        return T

    def calculate_utility(self, loc, discount, utility):
        if self.terminalMask[loc]:
            return self.reward[loc]
        row, col = loc
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, :, :] * utility,
                       axis=-1),
                axis=-1)
        ) + self.reward[loc]

    def plot_policy(self, utility, policy=None):
        if policy is None:
            policy = self.best_policy(utility)
        markers = "^>v<"
        marker_size = 200 // np.max(policy.shape)
        marker_edge_width = marker_size // 10
        marker_fill_color = 'w'

        no_action_mask = self.terminalMask | self.obstacleMask

        utility_normalized = (utility - utility.min()) / \
                             (utility.max() - utility.min())

        utility_normalized = (255*utility_normalized).astype(np.uint8)

        utility_rgb = cv2.applyColorMap(utility_normalized, cv2.COLORMAP_JET)
        for i in range(3):
            channel = utility_rgb[:, :, i]
            channel[self.obstacleMask] = 0

        plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')

        for i, marker in enumerate(markers):
            y, x = np.where((policy == i) &
                            np.logical_not(no_action_mask))
            plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

        y, x = np.where(self.terminalMask)
        plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
                 color=marker_fill_color)

        tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
        tick_step = np.max(policy.shape)/8
        best_option = np.argmin(
            np.abs(np.log(tick_step) - np.log(tick_step_options)))
        tick_step = tick_step_options[best_option]
        plt.xticks(np.arange(0, policy.shape[1] - 0.5, tick_step))
        plt.yticks(np.arange(0, policy.shape[0] - 0.5, tick_step))
        plt.xlim([-0.5, policy.shape[0]-0.5])
        plt.xlim([-0.5, policy.shape[1]-0.5])
