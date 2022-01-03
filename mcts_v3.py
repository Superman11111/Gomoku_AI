from io import SEEK_CUR
import random
import copy
import math as np
import time
import argparse
import json

from numpy.lib.function_base import piecewise


MAX_BOARD = 15

board = [[0 for ip in range(MAX_BOARD)] for j in range(MAX_BOARD)]


class Node:

    def __init__(self, move, parent=None,
                 num_child=-1, possible_moves_for_child=None, possible_moves_for_expansion=None,
                 board_width=None, board_height=None, num_expand=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.sim_num = 0
        self.win_num = 0
        if parent is None:
            self.max_num_child = num_child
            self.max_num_expansion = num_expand
            self.board_width = board_width
            self.board_height = board_height
            # inherit the possible moves for this node's children, similar to board.availables
            # so not used yet
            self.possible_moves_for_child = copy.deepcopy(possible_moves_for_child)

            self.possible_moves_for_expansion = copy.deepcopy(possible_moves_for_expansion)
            self.player = 1
            self.opponent = 2
        if parent is not None:
            if parent.max_num_child > 0:
                self.max_num_child = parent.max_num_child - 1
            # avoid expanding a move twice
            parent.possible_moves_for_expansion.remove(move)

            # independently inherit
            self.possible_moves_for_child = copy.deepcopy(parent.possible_moves_for_child)
            self.possible_moves_for_child.remove(move)
            # update the neighbor information
            self.possible_moves_for_expansion = copy.deepcopy(parent.possible_moves_for_expansion)
            self.board_width = parent.board_width
            self.board_height = parent.board_height
            x, y = move
            up, down, left, right = x, self.board_height - 1 - x, y, self.board_width - 1 - y
            if up:
                self.possible_moves_for_expansion.add((x - 1, y))
                if left:
                    self.possible_moves_for_expansion.add((x - 1, y - 1))
                if right:
                    self.possible_moves_for_expansion.add((x - 1, y + 1))
            if down:
                self.possible_moves_for_expansion.add((x + 1, y))
                if left:
                    self.possible_moves_for_expansion.add((x + 1, y - 1))
                if right:
                    self.possible_moves_for_expansion.add((x + 1, y + 1))
            if left:
                self.possible_moves_for_expansion.add((x, y - 1))
            if right:
                self.possible_moves_for_expansion.add((x, y + 1))
            self.possible_moves_for_expansion = self.possible_moves_for_expansion & self.possible_moves_for_child
            self.max_num_expansion = len(self.possible_moves_for_expansion)
            self.opponent = parent.player
            self.player = parent.opponent
            parent.children.append(self)



class Board:

    def __init__(self, input_board, n_in_line=5):
        assert type(n_in_line) == int, "n_in_line para should be INT!"
        self.width = MAX_BOARD
        self.height = MAX_BOARD
        self.board = copy.deepcopy(input_board)
        self.n_in_line = n_in_line
        self.availables = set([
            (i, j) for i in range(self.height) for j in range(self.width) if input_board[i][j] == 0
        ])
        self.neighbors = self.getNeighbors()

    def is_free(self, x, y):
        return 1 if self.board[x][y] == 0 else 0

    def getNeighbors(self):
        neighbors = set()
        if len(self.availables) == self.width * self.height:
            "if the board is empty, then choose from the center one"
            "assume our board is bigger than 1x1"
            x0, y0 = self.width // 2 - 1, self.height // 2 - 1
            neighbors.add((x0, y0))
            return neighbors
        else:
            for i in range(self.height):
                for j in range(self.width):
                    if self.board[i][j]:
                        neighbors.add((i - 1, j - 1))
                        neighbors.add((i - 1, j))
                        neighbors.add((i - 1, j + 1))
                        neighbors.add((i, j - 1))
                        neighbors.add((i, j + 1))
                        neighbors.add((i + 1, j - 1))
                        neighbors.add((i + 1, j))
                        neighbors.add((i + 1, j + 1))

                        neighbors.add((i - 2, j - 2))
                        neighbors.add((i - 2, j))
                        neighbors.add((i - 2, j + 2))
                        neighbors.add((i, j - 2))
                        neighbors.add((i, j + 2))
                        neighbors.add((i + 2, j - 2))
                        neighbors.add((i + 2, j))
                        neighbors.add((i + 2, j + 2))

            neighbors = neighbors & self.availables
            return neighbors

    def update(self, player, move, update_neighbor=True):
        """
        update the board and check if player wins, so one should use like this:
            if board.update(player, move):
        :param player: the one to take the move
        :param move: a tuple (x, y)
        :param update_neighbor: built for periods when you are sure no one wins
        :return: 1 denotes player wins and 0 denotes not
        """
        assert len(move) == 2, "move is invalid, length = {}".format(len(move))
        self.board[move[0]][move[1]] = player
        self.availables.remove(move)

        if update_neighbor:
            self.neighbors.remove(move)

            neighbors = set()
            x, y = move
            up, down, left, right = x, self.height - 1 - x, y, self.width - 1 - y
            if up:
                neighbors.add((x - 2, y))
                neighbors.add((x - 1, y))                
                if left:
                    neighbors.add((x - 1, y - 1))
                    neighbors.add((x - 2, y - 2))
                if right:
                    neighbors.add((x - 1, y + 1))
                    neighbors.add((x - 2, y + 2))
            if down:
                neighbors.add((x + 2, y))
                neighbors.add((x + 1, y))

                if left:
                    neighbors.add((x + 1, y - 1))
                    neighbors.add((x + 2, y - 2))
                if right:
                    neighbors.add((x + 1, y + 1))
                    neighbors.add((x + 2, y + 2))
            if left:
                neighbors.add((x, y - 2))
                neighbors.add((x, y - 1))
            if right:
                neighbors.add((x, y + 1))
                neighbors.add((x, y + 2))
            neighbors = self.availables & neighbors
            self.neighbors = self.neighbors | neighbors

    def getValue(self, length, open):
        if (length >= 5):
            return 2000
        
        if (length == 4 and open == 2):
            return 1600
        if (length == 4 and open == 1):
            return 600
        if (length == 3 and open == 2):
            return 600
        if (length == 3 and open == 1):
            return 50
        if (length == 2 and open == 2):
            return 5
        if (length == 2 and open == 1):
            return 4
        if (length == 1 and open == 2):
            return 2
        if (length == 1 and open == 1):
            return 1
        return 0
    
    def checkStatus(self, player, move) :
        self.board[move[0]][move[1]] = player
        x_this, y_this = move
        """x++, x--"""
        length = -1
        maxV = 0
        open = 0
        count = 0
        x_save = x_this
        y_save = y_this
        while (x_this >= 0 and self.board[x_this][y_this] == player):
            x_this -= 1
            length += 1

        if(x_this >= 0 and self.board[x_this][y_this] == 0):
            open += 1

        x_this = x_save

        while (x_this < self.height and self.board[x_this][y_this] == player):
            x_this += 1
            length += 1
        
        if (x_this < self.height and self.board[x_this][y_this] == 0):
            open += 1
        v = self.getValue(length, open)
        if(v > 1000):
            self.board[move[0]][move[1]] = 0
            return v
        if(v > 500): 
            count += 1
        if(maxV < v):
            maxV = v


        """y++, y--"""
        length = -1
        open = 0
        x_this = x_save
        y_this = y_save
        while (y_this >= 0 and self.board[x_this][y_this] == player):
            y_this -= 1
            length += 1

        if(y_this >= 0 and self.board[x_this][y_this] == 0):
            open += 1

        y_this = y_save

        while (y_this < self.width and self.board[x_this][y_this] == player):
            y_this += 1
            length += 1
        
        if (y_this < self.width and self.board[x_this][y_this] == 0):
            open += 1
        v = self.getValue(length, open)
        if(v > 1000):
            self.board[move[0]][move[1]] = 0
            return v
        if(v > 500): 
            count += 1
        if(maxV < v):
            maxV = v


        """x++, y++"""
        x_this = x_save
        y_this = y_save
        length = -1
        open = 0
        while (x_this < self.height and  y_this < self.width and self.board[x_this][y_this] == player):
            y_this += 1
            x_this += 1
            length += 1

        if(x_this <self.height and y_this < self.width and self.board[x_this][y_this] == 0):
            open += 1

        y_this = y_save
        x_this = x_save
        while (x_this >= 0 and y_this >= 0 and self.board[x_this][y_this] == player):
            y_this -= 1
            x_this -= 1
            length += 1
        
        if (x_this >= 0 and y_this >= 0 and self.board[x_this][y_this] == 0):
            open += 1
        v = self.getValue(length, open)
        if(v > 1000):
            self.board[move[0]][move[1]] = 0
            return v
        if(v > 500): 
            count += 1
        if(maxV < v):
            maxV = v


        """x++, y--"""
        y_this = y_save
        x_this = x_save
        length = -1
        open = 0
        while (x_this < self.height and  y_this >= 0 and self.board[x_this][y_this] == player):
            y_this -= 1
            x_this += 1
            length += 1

        if(x_this < self.height and y_this >= 0 and self.board[x_this][y_this] == 0):
            open += 1

        y_this = y_save
        x_this = x_save

        while (x_this >= 0 and y_this < self.width and self.board[x_this][y_this] == player):
            y_this += 1
            x_this -= 1
            length += 1
        
        if (y_this < self.width and x_this >= 0  and self.board[x_this][y_this] == 0):
            open += 1
        v = self.getValue(length, open)
        if(v > 1000):
            self.board[move[0]][move[1]] = 0
            return v
        if(v > 500): 
            count += 1
        if(maxV < v):
            maxV = v


        y_this = y_save
        x_this = x_save

        if(x_this - 4 > 0 and x_this + 1 < self.height and self.board[x_this - 1][y_this] == 0 and self.board[x_this - 2][y_this] == player and 
            self.board[x_this - 3][y_this] == player and self.board[x_this - 4][y_this] == 0 and 
            self.board[x_this + 1][y_this] == 0):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(y_this - 4 > 0 and y_this + 1 < self.width and self.board[x_this][y_this - 1] == 0 and self.board[x_this][y_this - 2] == player and 
            self.board[x_this][y_this - 3] == player and self.board[x_this][y_this - 4] == 0 and 
            self.board[x_this][y_this + 1] == 0):
            v = 450 
            count += 1
            if(maxV < v):
                maxV = v


        if(x_this + 4 < self.height and x_this - 1 > 0 and self.board[x_this + 1][y_this] == 0 and self.board[x_this + 2][y_this] == player and 
            self.board[x_this + 3][y_this] == player and self.board[x_this + 4][y_this] == 0 and 
            self.board[x_this - 1][y_this] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(y_this + 4 < self.height and y_this -1 > 0 and self.board[x_this][y_this + 1] == 0 and self.board[x_this][y_this + 2] == player and 
            self.board[x_this][y_this + 3] == player and self.board[x_this][y_this + 4] == 0 and 
            self.board[x_this][y_this - 1] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v


        if(x_this - 2 > 0 and x_this + 3 < self.height and self.board[x_this - 1][y_this] == player and self.board[x_this - 2][y_this] == 0 and 
            self.board[x_this + 1][y_this] == 0 and self.board[x_this + 2][y_this] == player and
            self.board[x_this + 3][y_this] == 0 ):
            v = 450
            
            count += 1
            if(maxV < v):
                maxV = v

        if(y_this - 2 > 0 and y_this + 3 < self.width and self.board[x_this][y_this - 1] == player and self.board[x_this][y_this - 2] == 0 and 
            self.board[x_this][y_this + 1] == 0 and self.board[x_this][y_this + 2] == player and
            self.board[x_this][y_this + 3] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this + 2 < self.height and x_this - 3 > 0 and self.board[x_this + 1][y_this] == player and self.board[x_this + 2][y_this] == 0 
            and self.board[x_this - 1][y_this] == 0 and self.board[x_this - 2][y_this] == player and
            self.board[x_this - 3][y_this] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(y_this - 3 > 0 and y_this + 2 < self.width and self.board[x_this][y_this + 1] == player and self.board[x_this][y_this + 2] == 0 
            and self.board[x_this][y_this - 1] == 0 and self.board[x_this][y_this - 2] == player and
            self.board[x_this][y_this - 3] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v



        if(x_this - 4 > 0 and y_this - 4 > 0 and x_this + 1 < self.height and y_this + 1 < self.width and
             self.board[x_this - 1][y_this - 1] == 0 and self.board[x_this - 2][y_this - 2] == player and 
            self.board[x_this - 3][y_this - 3] == player and self.board[x_this - 4][y_this - 4] == 0 and 
            self.board[x_this + 1][y_this + 1] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this - 1 > 0 and y_this - 4 > 0 and x_this + 4 < self.height and y_this + 1 < self.width and
            self.board[x_this + 1][y_this - 1] == 0 and self.board[x_this + 2][y_this - 2] == player and 
            self.board[x_this + 3][y_this - 3] == player and self.board[x_this + 4][y_this - 4] == 0 and 
            self.board[x_this - 1][y_this + 1] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this - 1 > 0 and y_this - 1 > 0 and x_this + 4 < self.height and y_this + 4 < self.width and
            self.board[x_this + 1][y_this + 1] == 0 and self.board[x_this + 2][y_this + 2] == player and 
            self.board[x_this + 3][y_this + 3] == player and self.board[x_this + 4][y_this + 4] == 0 and 
            self.board[x_this - 1][y_this - 1] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this - 4 > 0 and y_this - 1 > 0 and x_this + 1 < self.height and y_this + 4 < self.width and
            self.board[x_this - 1][y_this + 1] == 0 and self.board[x_this - 2][y_this + 2] == player and 
            self.board[x_this - 3][y_this + 3] == player and  self.board[x_this - 4][y_this + 4] == 0 and
            self.board[x_this + 1][y_this - 1] == 0 ):
            v = 450
            count += 1   
            if(maxV < v):
                maxV = v


        if(x_this + 2 < self.height and y_this + 2 < self.width and x_this - 3 > 0 and y_this - 3 > 0 and
            self.board[x_this - 1][y_this - 1] == 0 and self.board[x_this - 2][y_this - 2] == player and 
            self.board[x_this + 1][y_this + 1] == player and self.board[x_this - 3][y_this - 3] == 0 and
            self.board[x_this + 2][y_this + 2] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this + 3 < self.height and y_this + 3 < self.width and x_this - 2 > 0 and y_this - 2 > 0 and
            self.board[x_this + 1][y_this + 1] == 0 and self.board[x_this + 2][y_this + 2] == player and 
            self.board[x_this - 1][y_this - 1] == player and self.board[x_this + 3][y_this + 3] == 0 and
            self.board[x_this - 2][y_this - 2] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this + 3 < self.height and y_this + 2 < self.width and x_this - 2 > 0 and y_this - 3 > 0 and
            self.board[x_this + 1][y_this - 1] == 0 and self.board[x_this + 2][y_this - 2] == player
            and self.board[x_this - 1][y_this + 1] == player and self.board[x_this + 3][y_this - 3] == 0 and
            self.board[x_this - 2][y_this + 2] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v

        if(x_this + 2 < self.height and y_this + 3 < self.width and x_this - 3 > 0 and y_this - 2 > 0 and
            self.board[x_this - 1][y_this + 1] == 0 and self.board[x_this + 2][y_this - 2] == player 
            and self.board[x_this + 1][y_this - 1] == player and self.board[x_this - 3][y_this + 3] == 0 and
            self.board[x_this + 2][y_this - 2] == 0 ):
            v = 450
            count += 1
            if(maxV < v):
                maxV = v



        self.board[move[0]][move[1]] = 0
        if(count >= 2):
            return 1200
        return maxV

    def check_win(self, player, move):
        """check if player win, this function will not actually do the move"""
        original = self.board[move[0]][move[1]]
        self.board[move[0]][move[1]] = player
        x_this, y_this = move
        # get the boundaries
        up = min(x_this, self.n_in_line - 1)
        down = min(self.height - 1 - x_this, self.n_in_line - 1)
        left = min(y_this, self.n_in_line - 1)
        right = min(self.width - 1 - y_this, self.n_in_line - 1)
        # \
        up_left = min(up, left)
        down_right = min(down, right)
        for i in range(up_left + down_right - self.n_in_line + 2):
            a = [
                self.board[x_this - up_left + i + j][y_this - up_left + i + j] for j in range(self.n_in_line)
            ]
            assert len(a) == self.n_in_line, "error when check if win on board"
            if len(set(a)) == 1 and a[0] > 0:
                self.board[move[0]][move[1]] = original
                return 1
        # /
        up_right = min(up, right)
        down_left = min(down, left)
        for i in range(up_right + down_left - self.n_in_line + 2):
            a = [
                self.board[x_this - up_right + i + j][y_this + up_right - i - j] for j in range(self.n_in_line)
            ]
            assert len(a) == self.n_in_line, "error when check if win on board"
            if len(set(a)) == 1 and a[0] > 0:
                self.board[move[0]][move[1]] = original
                return 1
        # --
        for i in range(left + right - self.n_in_line + 2):
            a = [
                self.board[x_this][y_this - left + i + j] for j in range(self.n_in_line)
            ]
            assert len(a) == self.n_in_line, "error when check if win on board"
            if len(set(a)) == 1 and a[0] > 0:
                self.board[move[0]][move[1]] = original
                return 1
        # |
        for i in range(up + down - self.n_in_line + 2):
            a = [
                self.board[x_this - up + i + j][y_this] for j in range(self.n_in_line)
            ]
            assert len(a) == self.n_in_line, "error when check if win on board"
            if len(set(a)) == 1 and a[0] > 0:
                self.board[move[0]][move[1]] = original
                return 1
        # no one wins
        self.board[move[0]][move[1]] = original
        return 0


class MCTS:

    def __init__(self, input_board, n_in_line=5, time_limit=5.0, max_simulation=5, max_simulation_one_play=50):
        self.time_limit = float(time_limit)
        self.max_simulation = max_simulation
        self.max_simulation_one_play = max_simulation_one_play
        self.MCTSboard = Board(input_board, n_in_line)    
        self.player = 2
        self.candidates = set()
        self.allmoves = dict()
        self.flag = 0
        for move in self.MCTSboard.neighbors:
            value1 = self.MCTSboard.checkStatus(1, move)
            value2 = self.MCTSboard.checkStatus(2, move)
            maxVal = max(value2 + 150, value1)
            if(maxVal > 2000):
                if(self.flag < 9):                 
                    self.candidates.clear()
                    self.flag = 9
                self.candidates.add(move)
                break            
            if(maxVal > 1800 and self.flag <= 8):
                if(self.flag < 8):
                    self.candidates.clear()
                    self.flag = 8
                self.candidates.add(move)
                continue
            if(maxVal > 1700 and self.flag <= 7):
                if(self.flag < 7):
                    self.candidates.clear()
                    self.flag = 7
                self.candidates.add(move)
                continue
            if(maxVal > 1500 and self.flag <= 6):
                if(self.flag < 6):
                    self.candidates.clear()
                    self.flag = 6
                self.candidates.add(move)
                continue
            if(maxVal > 1300 and self.flag <= 5):
                if(self.flag < 5):
                    self.candidates.clear()
                    self.flag = 5
                self.candidates.add(move)
                continue
            if(maxVal > 1100 and self.flag <= 4):
                if(self.flag < 4):
                    self.candidates.clear()
                    self.flag = 4
                self.candidates.add(move)
                continue
            if(maxVal > 700 and self.flag <= 3):
                if(self.flag < 3):
                    self.candidates.clear()
                    self.flag = 3
                self.candidates.add(move)
                continue
            if(maxVal > 600 and self.flag <= 2):
                if(self.flag < 2):
                    self.candidates.clear()
                    self.flag = 2
                self.candidates.add(move)
                continue
            self.allmoves[move] = maxVal
        if(self.flag == 0):
            orders = sorted(self.allmoves.items(), key = lambda x:x[1],reverse = True)
            if(len(orders) < 9):
                for iter in range(len(orders)):
                    if( (orders[iter][1] > 152 or 5 < orders[iter][1] < 150) and orders[iter][1] != 450):
                        self.candidates.add(orders[iter][0])
                if(len(self.candidates) == 0):
                    for iter in range(len(orders)):
                        self.candidates.add(orders[iter][0])
            else:
                for iter in range(9):
                    if((orders[iter][1] > 152 or 5 < orders[iter][1] < 150) and orders[iter][1] != 450):
                        self.candidates.add(orders[iter][0])
                if(len(self.candidates) == 0):
                    for iter in range(9):
                        self.candidates.add(orders[iter][0])
        self.root = Node(None,
                         parent=None,
                         num_child=len(self.MCTSboard.availables),
                         possible_moves_for_child=self.MCTSboard.availables,
                         possible_moves_for_expansion=self.candidates,
                         num_expand=len(self.candidates),
                         board_width=self.MCTSboard.width,
                         board_height=self.MCTSboard.height)
        self.get_player = {
            1: 2,
            2: 1,
        }

    def get_action(self):
        if len(self.MCTSboard.availables) == 1:
            return list(self.MCTSboard.availables)[0] 

        num_nodes = 0
        begin_time = time.time()
        while time.time() - begin_time < self.time_limit:
            node_to_expand = self.select_and_expand()
            if(node_to_expand == None):
                continue
            for _ in range(self.max_simulation):
                board_deep_copy = copy.deepcopy(self.MCTSboard)
                if(time.time() - begin_time > self.time_limit):
                    break
                self.simulate_and_bp(board_deep_copy, node_to_expand)
            num_nodes += 1

        percent_wins, move = max(
            (child.win_num / child.sim_num , child.move)
            for child in self.root.children
        ) 

        return move

    def select_and_expand(self):
        "Selection: greedy search based on UCB value"
        currentNode = self.root
        while currentNode.children:
            if len(currentNode.children) < currentNode.max_num_expansion:
                break

            ucb, selectedNode = 0, None
            for child in currentNode.children:

                childUCB = child.win_num / child.sim_num + np.sqrt(
                    2 * np.log(currentNode.sim_num) / child.sim_num
                ) 
                if childUCB >= ucb:
                    ucb, selectedNode = childUCB, child
            currentNode = selectedNode
        "Expansion: randomly expand a node"
        if(len(list(currentNode.possible_moves_for_expansion)) == 0):
            return None
        expandMove = random.choice(list(currentNode.possible_moves_for_expansion))
        expandNode = Node(expandMove, parent=currentNode)
        return expandNode

        
    def simulate_and_bp(self, cur_board, expandNode):
        _node = expandNode

        while _node.parent.move:
            _node = _node.parent
            cur_board.update(_node.player, _node.move, update_neighbor=False)

        if len(cur_board.neighbors) == 0:
            return

        cur_board.neighbors = cur_board.getNeighbors()
        player = expandNode.player
        win = cur_board.check_win(player, expandNode.move)
        cur_board.update(player, expandNode.move)
        for t in range(1, self.max_simulation_one_play + 1):
            is_full = not len(cur_board.neighbors)
            if win or is_full:
                break
            player = self.get_player[player]
            move = random.choice(list(cur_board.neighbors))
            win = cur_board.check_win(player, move)
            cur_board.update(player, move)

        currentNode = expandNode
        while currentNode:
            currentNode.sim_num += 1
            if win and player == currentNode.player == expandNode.player:
                currentNode.win_num += 1
            currentNode = currentNode.parent
 

def isFree(x, y):
    return 0 <= x < MAX_BOARD and 0 <= y < MAX_BOARD and board[x][y] == 0


if __name__ == "__main__":

    full_input = json.loads(input())
    all_requests = full_input["requests"]
    all_responses = full_input["responses"]
    if(all_requests[0]["x"] == -1 and len(all_responses) == 0):
        my_action = { "x": int(7), "y": 7 }
        print(json.dumps({
            "response": my_action,
            "data": None
        }))
    else:
        for i in range(len(all_requests)):
            if(len(all_requests) > len(all_responses) and i == len(all_responses)):
                oppX = int(all_requests[i]["x"]) 
                oppY = int(all_requests[i]["y"])
                board[oppX][oppY] = 1
                break
            myX = int(all_responses[i]["x"]) 
            myY = int(all_responses[i]["y"])
            board[myX][myY] = 2
            if(all_requests[i]["x"] == -1):
                continue
            oppX = int(all_requests[i]["x"]) 
            oppY = int(all_requests[i]["y"])
            board[oppX][oppY] = 1 
       
        MCTS_AI = MCTS(board,
                    n_in_line=5,
                    time_limit=5.4,
                    max_simulation=150,  
                    max_simulation_one_play=120)
        while True:
            move = MCTS_AI.get_action()
            x, y = move
            if isFree(x, y):
                break
        my_action = { "x": int(x), "y": y }
        print(json.dumps({
            "response": my_action,
            "data": None
        }))