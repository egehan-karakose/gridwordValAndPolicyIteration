from QLearning import QLearning
from ValueIter import *
from PolicyIter import *

import numpy as np
import matplotlib.pyplot as plt

# functions to editing results and plots
# ----------------------------------------


def direction_changer(policy_grid, i):
    corrected_policy_grid = policy_grid
    if i == 1:
        return (policy_grid)
    else:
        for i in range(4):
            for j in range(4):
                if policy_grid[i, j] == 2:
                    corrected_policy_grid[i, j] = 1
                elif policy_grid[i, j] == 1:
                    corrected_policy_grid[i, j] = 2

        return corrected_policy_grid


def plot_convergence(utility_grids, policy_grids):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility', color='b')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
    ax2.plot(policy_changes, 'r.-')
    ax2.set_ylabel('Change in Best Policy', color='r')


def run(solver_name, solver_fn, solver, iteration, discount):
    print('Final result of {}:'.format(solver_name))
    print('                           ')
    print(" s0     s1     s2     s3     s4    s5     s6     s7    s9     s10")
    policy_grids, utility_grids = solver_fn(
        iterations=iteration, discount=discount)  # d
    print('                           ')
    print(direction_changer(policy_grids[:, :, -1], 0))
    print('                           ')
    print(utility_grids[:, :, -1])
    print('                           ')
    print('-------------------------------------------------------------------------')
    print('                           ')
    plt.figure()
    solver.plot_policy(utility_grids[:, :, -1])
    plot_convergence(utility_grids, policy_grids)
    plt.show()


# ----------------------------------------
# Implemetation of gridworld and variables
# ----------------------------------------
if __name__ == '__main__':
    shape = (4, 4)
    goal = (0, -1)
    goal2 = (3, 1)
    goal3 = (3, -1)
    trap = (3, 2)
    obstacle = (1, 1)
    start = (2, 0)
    default_reward = 0  # r
    goal_reward = 1
    goal_reward2 = 10
    trap_reward = -10

    reward_grid = np.zeros(shape) + default_reward
    reward_grid[goal] = goal_reward
    reward_grid[goal2] = goal_reward
    reward_grid[goal3] = goal_reward2
    reward_grid[trap] = trap_reward
    reward_grid[obstacle] = 0

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    terminal_mask[goal] = True
    terminal_mask[goal2] = True
    terminal_mask[goal3] = True
    terminal_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
    obstacle_mask[1, 1] = True

# ----------------------------------------
    # value iteration and policy iteration
# ----------------------------------------


def Iters(p, d):
    valIter = ValueIter(reward_grid=reward_grid,
                        obstacle_mask=obstacle_mask,
                        terminal_mask=terminal_mask,
                        action_probabilities=[
                            (-1, (1-p)/2),  # left
                            (0, p),  # up
                            (1, (1-p)/2),  # right
                        ],
                        no_action_probability=0.0
                        )

    policyIter = PolicyIter(reward_grid=reward_grid,
                            obstacle_mask=obstacle_mask,
                            terminal_mask=terminal_mask,
                            action_probabilities=[
                                (-1, (1-p)/2),  # left
                                (0, p),  # up
                                (1, (1-p)/2),  # right
                            ],
                            no_action_probability=0.0
                            )

    run('Value Iteration', valIter.run_value_iterations,
        valIter, iteration=20, discount=d)
    run('Policy Iteration', policyIter.run_policy_iterations,
        policyIter, iteration=20, discount=d)


# ----------------------------------------
    # Q - Learning
# ----------------------------------------

def QLearn(a, d, e, N):

    gw = GridWorldMDP(reward_grid=reward_grid,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      action_probabilities=[
                          (-1, 0),  # left
                          (0, 1),  # up
                          (1, 0),  # right
                      ],
                      no_action_probability=0.0
                      )
    ql = QLearning(num_states=(shape[0] * shape[1]),
                   num_actions=4,
                   learning_rate=a,  # a
                   discount_rate=d,  # d
                   exploration_prob=e  # e
                   )

    start_state = gw.grid_coordinates_to_indices(start)

    iterations = N
    flat_policies, flat_utilities = ql.learn(start_state,
                                             gw.generate_experience,
                                             iterations=iterations)  # N

    new_shape = (gw.shape[0], gw.shape[1], iterations)
    ql_utility_grids = flat_utilities.reshape(new_shape)
    ql_policy_grids = flat_policies.reshape(new_shape)
    print('Final result of QLearning:')
    print('                           ')
    print(direction_changer(ql_policy_grids[:, :, -1], 1))
    direction_changer(ql_policy_grids[:, :, -1], 0)
    print('                           ')
    print(ql_utility_grids[:, :, -1])

    plt.figure()
    gw.plot_policy(ql_utility_grids[:, :, -1], ql_policy_grids[:, :, -1])
    plot_convergence(ql_utility_grids, ql_policy_grids)
    plt.show()
# ----------------------------------------


# REPORT

print("-------------REPORT------------")
print('                           ')

# Optimal Policy( R: Right, D: Down, Up: U, Left: L ) VI and PI

# U     U     U      1
# U     O     U      R
# U     U     U      U
# U     1    -10    10

Iters(p=1, d=1)  # to change reward change default_reward

# Optimal Policy(R: Right, D: Down, Up: U, Left: L) for Q-Learning

#    L      D     D      1
#    R      O     D      R
#    D      U     U      D
#    U      1    -10     10

QLearn(a=0.1, d=1, e=0, N=1000)

# Increasing N does not change value because it is already converge approximately at 20th iteration.


# a -------------------------------------
# For VI and PI policy and values are change it is getting better that r = 0. Because after that for each action it is getting - 0.01 reward, because of this it need to reach finish point with less action.


print("------------- Question a  ------------")
print('                           ')
default_reward = -0.01
reward_grid = np.zeros(shape) + default_reward
reward_grid[goal] = goal_reward
reward_grid[goal2] = goal_reward
reward_grid[goal3] = goal_reward2
reward_grid[trap] = trap_reward
reward_grid[obstacle] = 0

terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
terminal_mask[goal] = True
terminal_mask[goal2] = True
terminal_mask[goal3] = True
terminal_mask[trap] = True

obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
obstacle_mask[1, 1] = True

Iters(p=1, d=1)  # to change reward change default_reward


# For Q-Learning values converge later.(Approximately = 40 )
# And Optimal Policy change because of reward is does not change(discount factor=1) so after many iteration
# reward will greater than gained reward so it must reach finish point sooner.


QLearn(a=0.1, d=1, e=0, N=1000)
# a -------------------------------------

# b -------------------------------------
print("------------- Question  b  ------------")
print('                           ')
# It does not change optimal policy but it change convergence point it getting sooner because since the whole # algorithm is about
# making decisions where the outcome partly depends on random inputs which can drift away over time,
# invalidating initial decision, it makes sense to prefer decisions which a better as short-term solutions.


Iters(p=1, d=0.2)  # to change reward change default_reward


# b -------------------------------------

# c  -------------------------------------
print("------------- Question c  ------------")
print('                           ')

# -	For VI getting if we say reward for each action is 5 and never decrease, it want to take highest reward and some finish reward is already fewer than action reward so it want to make action instead of reaching finish. Optimal Policy is same as r = 1 because of same reason.

# -	For PI, policy does not updated after actions with same reason with VI.
# It is same with how it was created.


default_reward = 5
reward_grid = np.zeros(shape) + default_reward
reward_grid[goal] = goal_reward
reward_grid[goal2] = goal_reward
reward_grid[goal3] = goal_reward2
reward_grid[trap] = trap_reward
reward_grid[obstacle] = 0

terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
terminal_mask[goal] = True
terminal_mask[goal2] = True
terminal_mask[goal3] = True
terminal_mask[trap] = True

obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
obstacle_mask[1, 1] = True
Iters(p=1, d=1)  # to change reward change default_reward


# c  -------------------------------------


# d ---------------------------------------
print("------------- Question d  ------------")
print('                           ')

# With this changes agent can move different directions with probabilities
# up: 0.5,
# left and right: 0.25
# so it changes optimal policy.
# Optimal Policy(R: Right, D: Down, Up: U, Left: L)
#
# R      R     L      1
# U      O     D      D
# R      U     U      D
# L      1    -10    10


default_reward = -0.01
reward_grid = np.zeros(shape) + default_reward
reward_grid[goal] = goal_reward
reward_grid[goal2] = goal_reward
reward_grid[goal3] = goal_reward2
reward_grid[trap] = trap_reward
reward_grid[obstacle] = 0

terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
terminal_mask[goal] = True
terminal_mask[goal2] = True
terminal_mask[goal3] = True
terminal_mask[trap] = True

obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
obstacle_mask[1, 1] = True
Iters(p=0.5, d=1)  # to change reward change default_reward


# d ---------------------------------------

# e -------------------------------------------
print("------------- Question e  ------------")
print('                           ')
# ----------------------------------------------
# i **************
print("------------- Question e -> i  ------------")
print('                           ')

# For result of Q-learning even if the values in calculations change, end of calculations it will #converge very close values with N=100 result ,
# after 1000 iteration. Q values(nearly) and Policy same with N = 100.


Iters(p=0.8, d=0.9)
QLearn(a=0.1, d=0.9, e=0, N=1000)

# ii **************
print("------------- Question e -> ii  ------------")
print('                           ')
# If N = 10000 nothing will change in policy but some value can change with very small amount because it is already converged.

QLearn(a=0.1, d=0.9, e=0, N=10000)

# iii **************

print("------------- Question e -> iii  ------------")
print('                           ')

# If we change e = 0.1 it can do some random actions so values can not be best actions value.
# Results are not so good. If exploration probability is useful when grid is so large but if grid is small
# it cause policy problems.

QLearn(a=0.1, d=0.9, e=0.1, N=10000)

# iv **************
print("------------- Question e -> iv  ------------")
print('                           ')
# If we increase learning rate = 1 the new infromation calculated by algrithm totaly overrides the old #information
# so we can not do anything about old explorations. But it directly converge the result values and policy.


QLearn(a=1, d=0.9, e=0, N=10000)

# v **************
print("------------- Question e -> v  ------------")
print('                           ')

# When we take learning rate 1/count, for each pair count increase 1 it reduce learning rate and each #iteration value of
# new information will decrease override size on old information will decrease so it a good combination of new #learning and known information.

QLearn(a=1, d=0.9, e=0, N=10000)  # uncomment line 47 in QLearning

# vi **************
print("------------- Question e -> vi  ------------")
print('                           ')

# If we increase N = 100000 with e: 0.1 and same learning rate with above the result more accurate than #just e: 0.1
# because we have a good leaning rate and some e: 0.1 with 100000 iteration the result is converge in 100000 #iteration.

QLearn(a=1, d=0.9, e=0.1, N=100000)

# vii  **************
# I think is the best parameters is e = 0 and a = 0.1 for this grid world, but if we have larger number of #statement with
# different parameters or values etc. The best parameters would be last one.
