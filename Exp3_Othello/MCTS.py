from func_timeout import func_timeout, FunctionTimedOut
from copy import deepcopy
from Node import Node
import random
import board
import math
import sys


class MC:
    def tree_search(self, board, color):
        """ 
        实现UCT方法
        return: action
        board: 当前棋局
        color: 当前玩家
        """
        # 特殊情况：只有一种选择
        actions = list(board.get_legal_actions(color))
        if len(actions) == 1:
            return list(actions)[0]

        # 创建根节点
        curBoard = deepcopy(board)
        root = Node(curBoard, None, color, None)

        # 时间限制（每一步在60s以内）
        try:
            func_timeout(59, self.while_function, args=[root])
        except FunctionTimedOut:
            pass

        return self.best_child(root, math.sqrt(2), color).curAction

    def while_function(self, root):
        # four steps
        while True:
            # selection, expantion
            nodeExpand = self.tree_policy(root)
            # simulation
            reward = self.default_policy(nodeExpand.board, nodeExpand.color)
            # backpropagation
            self.back_propagation(nodeExpand, reward)

    def expand(self, node):
        """ 
        输入一节点，在该节点上拓展一个新的节点
        使用random方法执行action，返回新增节点
        """
        action = random.choice(node.unvisitedActions)
        node.unvisitedActions.remove(action)

        # 执行action，得到新的board
        newBoard = deepcopy(node.board)
        if action != "noRes":
            newBoard._move(action, node.color)
        else:
            pass

        newColor = 'X' if node.color == 'O' else 'O'
        newNode = Node(newBoard, node, newColor, action)
        node.children.append(newNode)
        return newNode

    def best_child(self, node, balance, color):
        # 计算每个子节点的best_val
        for child in node.children:
            child.calc_best_val(balance, color)

        # 对子节点按照best_val降序排序
        sortedChildren = sorted(node.children,
                                key=lambda x: x.bestVal[color],
                                reverse=True)

        # 返回best_val最大的元素
        return sortedChildren[0]

    def tree_policy(self, node):
        """
        传入当前需要开始搜索的节点（从根节点开始）
        返回最好的需要expend的节点
        叶子结点则直接返回
        """
        returnNode = node
        while not returnNode.isover:
            if len(returnNode.unvisitedActions) > 0:
                # 有未expand节点
                return self.expand(returnNode)
            else:
                # 选择val最大的
                returnNode = self.best_child(returnNode, math.sqrt(2),
                                             returnNode.color)

        return returnNode

    def default_policy(self, board, color):
        """
        输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward
        保证输入的节点有未执行的action可以expend，随机选择action
        """
        newBoard = deepcopy(board)
        newColor = color

        def game_over(board):
            l2 = list(board.get_legal_actions('O'))
            l1 = list(board.get_legal_actions('X'))
            return len(l2) == 0 and len(l1) == 0

        while not game_over(newBoard):
            actions = list(newBoard.get_legal_actions(newColor))
            if len(actions) == 0:
                action = None
            else:
                action = random.choice(actions)

            if action is None:
                pass
            else:
                newBoard._move(action, newColor)

            newColor = 'X' if newColor == 'O' else 'O'

        winner, diff = newBoard.get_winner()
        diff /= 64
        return winner, diff

    def back_propagation(self, node, reward):
        newNode = node
        while newNode is not None:
            newNode.visitedTimes += 1
            if reward[0] == 0:
                newNode.reward['X'] += reward[1]
                newNode.reward['O'] -= reward[1]
            elif reward[0] == 1:
                newNode.reward['X'] -= reward[1]
                newNode.reward['O'] += reward[1]
            elif reward[0] == 2:
                pass

            newNode = newNode.parent
