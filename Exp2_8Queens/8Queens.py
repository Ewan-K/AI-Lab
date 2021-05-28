import numpy as np  # 提供维度数组与矩阵运算
import copy  # 从copy模块导入深度拷贝方法
from board import Chessboard


# 基于棋盘类，设计搜索策略
class Game:
    def __init__(self, show=True):
        """
        初始化游戏状态.
        """

        self.chessBoard = Chessboard(show)
        self.solves = []
        self.gameInit()

    # 重置游戏
    def gameInit(self, show=True):
        """
        重置棋盘.
        """

        self.Queen_setRow = [-1] * 8
        self.chessBoard.boardInit(False)

    ##############################################################################
    ####              输出：self.solves = 八皇后所有序列解的list                ####
    ####             如:[[0,6,4,7,1,3,5,2],]代表八皇后的一个解为                ####
    ####           (0,0),(1,6),(2,4),(3,7),(4,1),(5,3),(6,5),(7,2)            ####
    ##############################################################################

    def collide(self, cur_queen, next_col):
        next_row = rows = len(cur_queen)
        for row in range(rows):
            column = cur_queen[row]
            if abs(column - next_col) in (0, next_row - row):
                return True
        return False

    def solve(self, num, cur_queen=[]):
        for pos in range(num):  # 八皇后的数量N=0, 1, 2, 3, 4, 5, 6 , 7 你要在哪一列放置皇后
            # 如果不冲突，则递归构造棋盘。
            if not self.collide(cur_queen, pos):  # 回溯法的体现
                # 如果棋盘状态state已经等于num-1，即到达倒数第二行，而这时最后一行皇后又没冲突，直接yield，打出其位置(pos, )
                if len(cur_queen) == num - 1:  # cur_queen=()
                    yield [
                        pos,
                    ]
                else:
                    for result in self.solve(num, cur_queen + [
                            pos,
                    ]):
                        yield [
                            pos,
                        ] + result

    def run(self, row=0):
        solutions = self.solve(8)
        self.solves = []
        for idx, solution in enumerate(solutions):
            self.solves.append(solution)

    def showResults(self, result):
        """
        结果展示.
        """

        self.chessBoard.boardInit(False)
        for i, item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i, item, False)

        self.chessBoard.printChessboard(False)

    def get_results(self):
        """
        输出结果(请勿修改此函数).
        return: 八皇后的序列解的list.
        """

        self.run()
        return self.solves


game = Game()
# game.chessBoard.play()
solutions = game.get_results()
print('There are {} results.'.format(len(solutions)))
for i in range(len(solutions)):
    print(solutions[i])
game.showResults(solutions[0])