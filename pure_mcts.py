import random
import copy
import math as np
import time
import argparse
import json

parser = argparse.ArgumentParser(description='DC-GAN on PyTorch')
parser.add_argument('--time-limit', default=5.0,
                    help='max time for one step', type=float)
parser.add_argument('--board-size', default=15,
                    help='board size, assuming the board is a square', type=int)
parser.add_argument('--n-in-line', default=5,
                    help='n=5 is the standard Gomoku', type=int)
parser.add_argument('--detail', default=False,
                    help='print the Nodes with higher than 0.4 winning rates', type=bool)
parser.add_argument('--max-simulation', default=20,
                    help='max simulation for one Node', type=int)
parser.add_argument('--max-simulation-one-step', default=150,
                    help='max simulation for one step, used for truncated simulation', type=int)
args = parser.parse_args()

MAX_BOARD = args.board_size
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]


class Node:

    def __init__(self, move, UCB=0, parent=None,
                 num_child=-1, possible_moves_for_child=None, possible_moves_for_expansion=None,
                 board_width=None, board_height=None, num_expand=None):
        self.move = move
        self.UCB = UCB  # not used yet
        self.parent = parent
        self.children = []
        self.sim_num = 0
        self.win_num = 0
        self.winner = 0
        if parent is None:
            self.max_num_child = num_child
            self.max_num_expansion = num_expand
            self.board_width = board_width
            self.board_height = board_height
            # inherit the possible moves for this node's children, similar to board.availables
            # so not used yet
            self.possible_moves_for_child = copy.deepcopy(possible_moves_for_child)

            self.possible_moves_for_expansion = copy.deepcopy(possible_moves_for_expansion)

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
            self.winner = parent.winner
        else:
            "zs: note that here is reverse because root is used to be your opponent's turn!!!"
            self.player = 1
            self.opponent = 2


class Board:

    def __init__(self, input_board, n_in_line=5):
        assert type(n_in_line) == int, "n_in_line para should be INT!"
        self.width = len(input_board[0])
        self.height = len(input_board)
        self.board = copy.deepcopy(input_board)
        self.n_in_line = n_in_line
        self.availables = set([
            (i, j) for i in range(self.height) for j in range(self.width) if input_board[i][j] == 0
        ])
        self.neighbors = self.get_neighbors()
        self.winner = None

    def is_free(self, x, y):
        return 1 if self.board[x][y] == 0 else 0

    def get_neighbors(self):
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

            return neighbors & self.availables

    def update(self, player, move, update_neighbor=True):
        """
        update the board and check if player wins, so one should use like this:
            if board.update(player, move):
                winner = board.winner
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
                neighbors.add((x - 1, y))
                if left:
                    neighbors.add((x - 1, y - 1))
                if right:
                    neighbors.add((x - 1, y + 1))
            if down:
                neighbors.add((x + 1, y))
                if left:
                    neighbors.add((x + 1, y - 1))
                if right:
                    neighbors.add((x + 1, y + 1))
            if left:
                neighbors.add((x, y - 1))
            if right:
                neighbors.add((x, y + 1))
            neighbors = self.availables & neighbors
            self.neighbors = self.neighbors | neighbors

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

    def __init__(self, input_board, n_in_line=5,
                 confidence=2.0, time_limit=5.0, max_simulation=5, max_simulation_one_play=50):
        self.time_limit = float(time_limit)
        self.max_simulation = max_simulation
        self.max_simulation_one_play = max_simulation_one_play
        self.MCTSboard = Board(input_board, n_in_line)     # a deep copy Board class object
        self.confidence = confidence                       # confidence level of exploration
        self.player = 2
        self.get_player = {
            1: 2,
            2: 1,
        }
        self.root = Node(None,
                         parent=None,
                         num_child=len(self.MCTSboard.availables),
                         possible_moves_for_child=self.MCTSboard.availables,
                         possible_moves_for_expansion=self.MCTSboard.neighbors,
                         num_expand=len(self.MCTSboard.neighbors),
                         board_width=self.MCTSboard.width,
                         board_height=self.MCTSboard.height)

    def get_action(self):
        if len(self.MCTSboard.availables) == 1:
            return list(self.MCTSboard.availables)[0]  # the only choice

        num_nodes = 0
        begin_time = time.time()
        while time.time() - begin_time < self.time_limit:
            # Selection & Expansion
            node_to_expand = self.select_and_expand()

            # Simulation & back propagation
            for _ in range(self.max_simulation):
                board_deep_copy = copy.deepcopy(self.MCTSboard)
                self.simulate_and_bp(board_deep_copy, node_to_expand)

            num_nodes += 1
        if args.detail:
            print("total nodes expanded in one action:{}".format(num_nodes))

        percent_wins, move = max(
            (child.win_num / child.sim_num + child.winner, child.move)
            for child in self.root.children
        )  # choose a move with highest winning rate
        if args.detail:
            for child in self.root.children:
                if child.win_num / child.sim_num > 0.4:
                    print(child.win_num / child.sim_num, child.move)
            print('=-'*20)
            print(percent_wins, move)

        return move

    def select_and_expand(self):
        "Selection: greedy search based on UCB value"
        cur_node = self.root
        while cur_node.children:
            # check if current node is fully expanded
            if len(cur_node.children) < cur_node.max_num_expansion:
                break

            ucb, select_node = 0, None
            for child in cur_node.children:

                ucb_child = child.win_num / child.sim_num + np.sqrt(
                    2 * np.log(cur_node.sim_num) / child.sim_num
                )
                if ucb_child >= ucb:
                    ucb, select_node = ucb_child, child
            cur_node = select_node

        "Expansion: randomly expand a node"
        expand_move = random.choice(list(cur_node.possible_moves_for_expansion))
        expand_node = Node(expand_move, parent=cur_node)
        return expand_node

    def simulate_and_bp(self, cur_board, expand_node):
        # first get to the board now
        _node = expand_node

        while _node.parent.move:
            _node = _node.parent
            cur_board.update(_node.player, _node.move, update_neighbor=False)

        "Simulation: do simulation randomly & neighborly"
        if len(cur_board.neighbors) == 0:
            return

        cur_board.neighbors = cur_board.get_neighbors()
        player = expand_node.player
        win = cur_board.check_win(player, expand_node.move)
        if win:
            expand_node.winner = player
        cur_board.update(player, expand_node.move)

        for t in range(1, self.max_simulation_one_play + 1):
            is_full = not len(cur_board.neighbors)
            if win or is_full:
                break

            player = self.get_player[player]
            move = random.choice(list(cur_board.neighbors))
            win = cur_board.check_win(player, move)
            cur_board.update(player, move)

        "Back propagation"
        cur_node = expand_node
        while cur_node:
            cur_node.sim_num += 1
            if win and cur_node.player == player:
                # print('--------', player)
                # for row in cur_board.board:
                #     print(' '.join([str(i) for i in row]))
                cur_node.win_num += 1
            cur_node = cur_node.parent






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
                    time_limit=5.6,
                    max_simulation=50,  
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