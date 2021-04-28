## Introduction

A python solution for 8Queens puzzle using DFS.

## Guideling

Running```
game.chessBoard.play()

````

will initiate a game interacting with user.
Running```
solutions = game.get_results()
print('There are {} results.'.format(len(solutions)))
for i in range(len(solutions)):
    print(solutions[i])
game.showResults(solutions[0])

````

will show all the possible solutions for 8Queens puzzle.
