import numpy as np
import random as rand
import xlwt
from xlwt import Workbook


class QLearning:
    '''A generic implementation of Q-Learning'''

    def __init__(self, *,
                 num_states,
                 num_actions,
                 learning_rate,
                 discount_rate,
                 exploration_prob
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_prob = exploration_prob

        self.savedPolicies = []
        self.savedUtilities = []

        # Initialize Q to small random values.
        np.random.seed(62)
        self._Q = np.zeros((num_states, num_actions), dtype=np.float)
        self._Q += np.random.normal(0, 0.3, self._Q.shape)

    def initialize(self, state):
        '''Set the initial state and return the learner's first action'''
        self.decide_next_action(state)
        self.stored_state = state
        return self.stored_action

    def learn(self, initial_state, experience_func, iterations=100):
        '''Iteratively experience new states and rewards'''
        all_policies = np.zeros((self.num_states, iterations))
        all_utilities = np.zeros_like(all_policies)
        count = 0
        for i in range(iterations):
            done = False
            self.initialize(initial_state)
            count += 1
            #self.learning_rate = 1 / count
            for j in range(iterations):
                state, reward, done = experience_func(self.stored_state,
                                                      self.stored_action)
                self.experience(state, reward)
                if done:
                    break

            policy, utility = self.get_policy_and_utility()
            all_policies[:, i] = policy
            all_utilities[:, i] = utility

            if i % 100 == 0:
                policy, utility = self.get_policy_and_utility()
                self.savedPlots(all_utilities, all_policies)
        print(self._Q)
        self.saveValues(self._Q)

        return all_policies, all_utilities

    # save plot for each 100 iteration
    def savedPlots(self, utility, policy):
        self.savedPolicies.append(policy)
        self.savedUtilities.append(utility)

    def experience(self, state, reward):
        '''The learner experiences state and receives a reward'''
        self.update_Q(self.stored_state, self.stored_action, state, reward)
        # determine an action and update the current state
        self.decide_next_action(state)
        self.stored_state = state

        return self.stored_action

    def get_policy_and_utility(self):
        policy = np.argmax(self._Q, axis=1)
        utility = np.max(self._Q, axis=1)
        return policy, utility

    def saveValues(self, value):
        wb = Workbook()
        sheet1 = wb.add_sheet('Sheet 1')

        s0 = value[0]
        s1 = value[1]
        s2 = value[2]
        s3 = value[4]
        s4 = value[6]
        s5 = value[7]
        s6 = value[8]
        s7 = value[9]
        s8 = value[10]
        s9 = value[11]
        s10 = value[12]

        s = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
        sheet1.write(0, 1, "s0")
        sheet1.write(0, 2, "s1")
        sheet1.write(0, 3, "s2")
        sheet1.write(0, 4, "s3")
        sheet1.write(0, 5, "s4")
        sheet1.write(0, 6, "s5")
        sheet1.write(0, 7, "s6")
        sheet1.write(0, 8, "s7")
        sheet1.write(0, 9, "s8")
        sheet1.write(0, 10, "s9")
        sheet1.write(0, 11, "s10")
        sheet1.write(1, 0, "up")
        sheet1.write(2, 0, "right")
        sheet1.write(3, 0, "down")
        sheet1.write(4, 0, "left")

        for i in range(len(s)):
            sheet1.write(1, i+1, str('{:.5f}'.format(s[i][0])))
            sheet1.write(2, i+1, str('{:.5f}'.format(s[i][1])))
            sheet1.write(3, i+1, str('{:.5f}'.format(s[i][2])))
            sheet1.write(4, i+1, str('{:.5f}'.format(s[i][3])))

        wb.save('QValues.xls')
        # update table

    def update_Q(self, s, a, s_prime, r):
        best_reward = self._Q[s_prime, self.find_best_action(s_prime)]
        self._Q[s, a] *= (1 - self.learning_rate)
        self._Q[s, a] += (self.learning_rate
                          * (r + self.discount_rate * best_reward))

    def decide_next_action(self, state):
        if rand.random() <= self.exploration_prob:
            self.stored_action = self.randomAction()
        else:
            self.stored_action = self.find_best_action(state)

    def randomAction(self):
        # decide the result of an action (straight if between 0
        # and 0.8, left if between 0.8 and 0.9 or right if between 0.9 and 1.0)
        a = rand.random()
        if a <= 0.8:
            return 1
        elif a <= 0.9 and a > 0.8:
            return 2
        elif a > 0.9:
            return 0

    def find_best_action(self, state):
        return int(np.argmax(self._Q[state, :]))
