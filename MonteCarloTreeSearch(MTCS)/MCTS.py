import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state          # The game state at this node
        self.parent = parent        # Parent node
        self.children = []          # Child nodes
        self.visits = 0             # Number of times node was visited
        self.value = 0              # Total value of the node

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_moves())

    def best_child(self, c_param=1.4):
        # UCB1 formula
        choices_weights = [
            (child.value / (child.visits + 1e-8)) + 
            c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        tried_moves = [child.state.last_move for child in self.children]
        possible_moves = self.state.get_possible_moves()

        for move in possible_moves:
            if move not in tried_moves:
                new_state = self.state.make_move(move)
                child_node = MCTSNode(new_state, parent=self)
                self.children.append(child_node)
                return child_node
        raise Exception("No moves to expand!")

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

    def is_terminal_node(self):
        return self.state.is_game_over()


def mcts(root_state, n_iter=1000):
    root_node = MCTSNode(root_state)

    for _ in range(n_iter):
        node = root_node

        # 1. Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child()

        # 2. Expansion
        if not node.is_terminal_node():
            node = node.expand()

        # 3. Simulation
        result = rollout(node.state)

        # 4. Backpropagation
        node.backpropagate(result)

    # Return the move leading to the best child
    best_move_node = max(root_node.children, key=lambda c: c.visits)
    return best_move_node.state.last_move


def rollout(state):
    current_state = state
    while not current_state.is_game_over():
        possible_moves = current_state.get_possible_moves()
        move = random.choice(possible_moves)
        current_state = current_state.make_move(move)
    return current_state.get_result()  # Return +1/-1/0 depending on win/loss/draw


# -----------------------------
# Tic-Tac-Toe Game State Class
# -----------------------------
class TicTacToe:
    def __init__(self, board=None, player=1, last_move=None):
        self.board = board if board else [0] * 9  # 0=empty, 1=X, -1=O
        self.player = player  # 1=X, -1=O
        self.last_move = last_move

    def get_possible_moves(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def make_move(self, move):
        new_board = self.board.copy()
        new_board[move] = self.player
        return TicTacToe(new_board, -self.player, last_move=move)

    def is_game_over(self):
        return self.get_winner() is not None or all(x != 0 for x in self.board)

    def get_result(self):
        winner = self.get_winner()
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        else:
            return 0

    def get_winner(self):
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns
            [0,4,8], [2,4,6]            # diagonals
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != 0:
                return self.board[line[0]]
        return None

    def print_board(self):
        symbols = {1: "X", -1: "O", 0: " "}
        for i in range(0, 9, 3):
            print("|".join(symbols[self.board[j]] for j in range(i, i+3)))
            if i < 6: print("-----")
        print()


# -----------------------------
# Play a game vs MCTS
# -----------------------------
def play_game():
    state = TicTacToe()
    while not state.is_game_over():
        state.print_board()
        if state.player == 1:
            # Human player
            move = int(input("Enter your move (0-8): "))
            state = state.make_move(move)
        else:
            # MCTS AI player
            print("AI thinking...")
            move = mcts(state, n_iter=1000)
            state = state.make_move(move)
            print(f"AI plays move {move}")

    state.print_board()
    result = state.get_result()
    if result == 1:
        print("X wins!")
    elif result == -1:
        print("O wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    play_game()