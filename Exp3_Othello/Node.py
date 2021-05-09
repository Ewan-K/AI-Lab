from math import log, sqrt


class Node:
    def __init__(self, board, parent, color, action):
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visitedTimes = 0  # 已访问次数
        self.board = board  # 选择这个Node时的棋盘
        self.color = color  # 当前玩家
        self.curAction = action  # 到达这个节点的action
        self.unvisitedActions = list(
            board.get_legal_actions(color))  # 未访问过的actions
        self.isover = self.game_over(board)  # 是否结束
        if (self.isover == False) and (len(self.unvisitedActions)
                                       == 0):  # 没步可走，游戏尚未结束
            self.unvisitedActions.append("noRes")

        self.bestVal = {'X': 0, 'O': 0}
        self.reward = {'X': 0, 'O': 0}

    def calc_best_val(self, balance, color):
        if self.visitedTimes == 0:
            print("-------------------------")
            print("visitedTimes = 0!")
            self.board.display()
            print("-------------------------")
        self.bestVal[
            color] = self.reward[color] / self.visitedTimes + balance * sqrt(
                2 * log(self.parent.visitedTimes) / self.visitedTimes)

    def game_over(self, board):
        l1 = list(board.get_legal_actions('X'))
        l2 = list(board.get_legal_actions('O'))
        return len(l1) == 0 and len(l2) == 0
