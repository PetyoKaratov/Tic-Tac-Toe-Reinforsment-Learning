import numpy as np
import pickle

# board dimensions
BOARD_ROWS = 3
BOARD_COLS = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.is_end = False
        self.board_hash = None
        # init p1 plays first
        self.player_symbol = 1

    def get_hash(self):
        """
        The function hashes the current board state so that it can be stored in the state-value dictionary.
        :return: unique hash of current board state
        """
        self.board_hash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.board_hash

    def available_positions(self):
        """
        Update the current vacant positions on the board
        :return: vacant positions as list of tuples
        """
        return [(i, j) for i in range(BOARD_ROWS)
                for j in range(BOARD_COLS)
                if self.board[i, j] == 0]

    def update_state(self, position):
        """
        Update position with player symbol
        and switch to another player
        :param position: player position
        """
        self.board[position] = self.player_symbol
        self.player_symbol = -1 if self.player_symbol == 1 else 1

    def winner(self):
        """
        After each action being taken by the player,
        continuously check if the game has ended and if ended ,
        judge the winner of the game and give reward to both players.
        :return: 1 if p1 wins, -1 if p2 wins, 0 if draw and None if the game is not yet ended
        """
        # check sum of row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.is_end = True
                return -1
        # check sum of col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.is_end = True
                return -1
        # check sum of diagonals and get abs max
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        # if abs max is 3 game is ended
        if diag_sum == 3:
            self.is_end = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.available_positions()) == 0:
            self.is_end = True
            return 0
        # not ended
        self.is_end = False
        return None

    # only when game ends give reward
    def give_reward(self):
        """
        At the end of game, 1 is rewarded to winner and 0 to loser.
        One thing to notice is that we consider draw is also a bad end,
        so we give our agent p1 0.1 reward (for player 1) even the game is tie
        (one can try out different reward to see how the agents act).
        """
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feed_reward(1)
            self.p2.feed_reward(0)
        elif result == -1:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)
        else:
            self.p1.feed_reward(0.1)
            self.p2.feed_reward(0.5)

    def reset(self):
        """
        board reset
        """
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board_hash = None
        self.is_end = False
        self.player_symbol = 1

    def play(self, rounds=100):
        """
        During training, the process for each player is:

            Look for available positions
            Choose action
            Update board state and add the action to player’s states
            Judge if reach the end of the game and give reward accordingly
        :param rounds: rounds to train on
        """
        # play rounds
        for i in range(1, rounds+1):
            # while game is not end
            while not self.is_end:
                # Player 1
                positions = self.available_positions()
                p1_action = self.p1.choose_action(positions, self.board, self.player_symbol)
                # take action and update board state
                self.update_state(p1_action)
                board_hash = self.get_hash()
                self.p1.add_state(board_hash)
                # check board status if it is ended

                win = self.winner()
                if win is not None:
                    # self.show_board()
                    # ended with p1 either win or draw
                    self.give_reward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.available_positions()
                    p2_action = self.p2.choose_action(positions, self.board, self.player_symbol)
                    self.update_state(p2_action)
                    board_hash = self.get_hash()
                    self.p2.add_state(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.give_reward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
            if i % 500 == 0:
                print(f"Rounds {i}")

    # play with human
    def play_with_human(self):
        # while game is not end
        while not self.is_end:
            # Player 1
            positions = self.available_positions()
            p1_action = self.p1.choose_action(positions, self.board, self.player_symbol)
            # take action and upate board state
            self.update_state(p1_action)
            self.show_board()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.available_positions()
                p2_action = self.p2.choose_action(positions)

                self.update_state(p2_action)
                self.show_board()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def evaluation_play(self, rounds=100):
        win_count = 0
        tie_count = 0
        for i in range(1, rounds + 1):
            # while game is not end
            while not self.is_end:
                # Player 1
                positions = self.available_positions()
                p1_action = self.p1.choose_action(positions, self.board, self.player_symbol)
                # take action and upate board state
                self.update_state(p1_action)
                self.show_board()
                # check board status if it is end
                win = self.winner()
                if win is not None:
                    if win == 1:
                        win_count += 1
                    else:
                        tie_count += 1
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.available_positions()
                    p2_action = self.p2.choose_action(positions, self.board, self.player_symbol)
                    # take action and upate board state
                    self.update_state(p2_action)
                    self.show_board()
                    # check board status if it is end
                    win = self.winner()
                    if win is not None:
                        self.reset()
                        break
        print(f'wins={win_count}')
        print(f'tie={tie_count}')
        print(f'lose={rounds - (win_count + tie_count)}')

    def show_board(self):
        # p1: X  p2: O
        for i in range(0, BOARD_ROWS):
            print('----+---+----')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'X'
                if self.board[i, j] == -1:
                    token = 'O'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----+---+----')


class Player:
    """
    Player agent
    """
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        # default exp_rate 0.3
        # (so 70% of the time our agent will take greedy action,
        # which is choosing action based on current estimation of states-value,
        # and 30% of the time our agent will take random action)
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # update the corresponding states (state -> value)

    @ staticmethod
    def get_hash(board):
        """
        The function hashes the current board state so that it can be stored in the state-value dictionary.
        :return: unique hash of current board state
        """
        board_hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return board_hash

    def choose_action(self, positions, current_board, symbol):
        """
        Store the hash of board state into state-value dict, and while exploitation,
        we hash the next board state and choose the action that returns the maximum value of next state.
        :param positions: list with positions
        :param current_board: array(3x3) with current board
        :param symbol: 1 or -1
        :return: Action (position to fill)
        """
        # select random or greedy action by using samples uniformly distributed over the half-open interval
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            # choosing action based on current estimation of states-value
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                # get hash of next board
                next_board_hash = self.get_hash(next_board)
                value = 0 if self.states_value.get(next_board_hash) is None else self.states_value.get(next_board_hash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    # append a hash state
    def add_state(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feed_reward(self, reward):
        """
        Update states values
        The updated value of state t equals the current value of state t adding the difference between
        the value of next state and the value of current state, which is multiplied by a learning rate α
        (Given the reward of intermediate state is 0)
        The positions of each game is stored in self.states and when the agent reach the end of the game,
        the estimates are updated in reversed fashion.
        :param reward: current reward
        :return: update reward
        """

        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        """
        Resset the state
        """
        self.states = []

    def save_policy(self):
        """
        Save policy file for future use to play with human player
        :return:
        """
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def load_policy(self, file):
        """
        Load the policy from file
        :param file: file path
        """
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def choose_action(positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action
            print('Invalid action. Try again.')


if __name__ == "__main__":
    # training
    # UNCOMMENT TO TRAIN NEW AGENT:
    # p1 = Player("p1")
    # p2 = Player("p2")
    #
    # st = State(p1, p2)
    # print("training...")
    # st.play(50000)
    # p1.save_policy()
    # evaluation play
    # p1 = Player("computer", exp_rate=0)
    # p1.load_policy("policy_p1")
    # p2 can be any trained player I used the p2
    # wins = 63
    # tie = 37
    # lose = 0

    # st = State(p1, p2)
    # st.evaluation_play()

    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.load_policy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play_with_human()
