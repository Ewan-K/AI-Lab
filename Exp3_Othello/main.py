# 导入黑白棋文件
from player import Player, RandomPlayer, HumanPlayer, AIPlayer
from board import Board
from game import Game
import random

# 人类玩家黑棋初始化
# black_player = AIPlayer("X", 1)
black_player = AIPlayer("X")

# AI玩家白棋初始化
# white_player = AIPlayer("O", 1)
white_player = AIPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()