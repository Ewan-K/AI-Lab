## Introduction

A python solution for 8Queens puzzle using DFS.

## Guideline

1. Running the line below in 8Queens.py will initiate a game interacting with user.

```
game.chessBoard.play()
```

2. Running the line below in 8Queens.py will show all the possible solutions for 8Queens puzzle.

```
solutions = game.get_results()
print('There are {} results.'.format(len(solutions)))
for i in range(len(solutions)):
print(solutions[i])
game.showResults(solutions[0])
```
