from Grid import *
import xlwt
from xlwt import Workbook


class ValueIter(GridWorldMDP):
    # Run Value iter with number of iterations
    # ----------------------------------------
    def run_value_iterations(self, discount=1.0,
                             iterations=10):
        utility_grids, policy_grids = self.init_utility_policy_storage(
            iterations)

        utility_grid = np.zeros_like(self.reward)
        wb = Workbook()
        sheet1 = wb.add_sheet('Sheet 1')
        for i in range(iterations):
            utility_grid = self.value_iteration(utility_grid=utility_grid)
            policy_grid = self.best_policy(utility_grid)
            utility_grids[:, :, i] = utility_grid
            policy_grids[:, :, i] = policy_grid

            s0 = str('{:.3f}'.format(utility_grid[0, 0]))
            s1 = str('{:.3f}'.format(utility_grid[0, 1]))
            s2 = str('{:.3f}'.format(utility_grid[0, 2]))
            s3 = str('{:.3f}'.format(utility_grid[1, 0]))
            s4 = str('{:.3f}'.format(utility_grid[1, 2]))
            s5 = str('{:.3f}'.format(utility_grid[1, 3]))
            s6 = str('{:.3f}'.format(utility_grid[2, 0]))
            s7 = str('{:.3f}'.format(utility_grid[2, 1]))
            s8 = str('{:.3f}'.format(utility_grid[2, 2]))
            s9 = str('{:.3f}'.format(utility_grid[2, 3]))
            s10 = str('{:.3f}'.format(utility_grid[3, 0]))
            print(s0 + " " + s1 + " " + s2 + " " + s3 + " " + s4 + " " + s5 + " "+s6 + " " + s7 + " "
                  + s8 + " " + s9 + " " + s10)
            sheet1.write(i, 0, s0)
            sheet1.write(i, 1, s1)
            sheet1.write(i, 2, s2)
            sheet1.write(i, 3, s3)
            sheet1.write(i, 4, s4)
            sheet1.write(i, 5, s5)
            sheet1.write(i, 6, s6)
            sheet1.write(i, 7, s7)
            sheet1.write(i, 8, s8)

        wb.save('valueIter.xls')
        return policy_grids, utility_grids

    # ----------------------------------------
    def value_iteration(self, utility_grid, discount=1.0):
        out = np.zeros_like(utility_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self.calculate_utility((i, j),
                                                   discount,
                                                   utility_grid)

        return out
