from Grid import *
import xlwt
from xlwt import Workbook


class PolicyIter(GridWorldMDP):
    # Run Policy iter with number of iterations
    # ----------------------------------------
    def run_policy_iterations(self, discount=1.0,
                              iterations=10):

        utilities, policies = self.init_utility_policy_storage(
            iterations)
        np.random.seed(62)
        policy = np.random.randint(0, self.numActions,
                                   self.shape)
        utility = self.reward.copy()
        wb = Workbook()
        sheet1 = wb.add_sheet('Sheet 1')

        for i in range(iterations):
            s0 = str('{:.3f}'.format(utility[0, 0]))
            s1 = str('{:.3f}'.format(utility[0, 1]))
            s2 = str('{:.3f}'.format(utility[0, 2]))
            s3 = str('{:.3f}'.format(utility[1, 0]))
            s4 = str('{:.3f}'.format(utility[1, 2]))
            s5 = str('{:.3f}'.format(utility[1, 3]))
            s6 = str('{:.3f}'.format(utility[2, 0]))
            s7 = str('{:.3f}'.format(utility[2, 1]))
            s8 = str('{:.3f}'.format(utility[2, 2]))
            s9 = str('{:.3f}'.format(utility[2, 3]))
            s10 = str('{:.3f}'.format(utility[3, 0]))

            print(s0 + " " + s1 + " " + s2 + " " + s3 + " " + s4 + " " + s5 + " "+s6 + " " + s7 + " "
                  + s8 + " " + s9 + " " + s10)

            policy, utility = self.policy_iteration(
                policy=policy,
                utility=utility
            )

            policies[:, :, i] = policy
            utilities[:, :, i] = utility
            sheet1.write(i, 0, s0)
            sheet1.write(i, 1, s1)
            sheet1.write(i, 2, s2)
            sheet1.write(i, 3, s3)
            sheet1.write(i, 4, s4)
            sheet1.write(i, 5, s5)
            sheet1.write(i, 6, s6)
            sheet1.write(i, 7, s7)
            sheet1.write(i, 8, s8)

        wb.save('policyIter.xls')
        return policies, utilities

    def policy_iteration(self, *, utility, policy, discount=1.0):
        r, c = self.grid_indices_to_coordinates()
        M, N = self.shape

        utility = (
            self.reward +
            discount * ((utility.reshape((1, 1, 1, M, N)) * self._T)
                        .sum(axis=-1).sum(axis=-1))[r, c, policy.flatten()]
            .reshape(self.shape)
        )

        for i in range(M):
            for j in range(N):
                utility[i, j] = self.calculate_utility((i, j),
                                                       discount,
                                                       utility)

        utility[self.terminalMask] = self.reward[self.terminalMask]

        return self.best_policy(utility), utility
