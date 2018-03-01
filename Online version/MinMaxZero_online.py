import time
round_start = time.time()

import json
import tensorflow as tf
import itertools
import numpy as np
from math import sqrt, log, exp
from numpy import unravel_index
from random import choice, random, sample
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, load_model
import keras.backend as K
K.set_image_dim_ordering('th')

C_T = 0
S_T = 0

class Ataxx:
    def __init__(self, board=None):
        if board is None:                  # if there is no initialization given
            self.data = np.zeros((7, 7), dtype=np.int8)   # then generate a board with starting init, and black(-1) takes first turn
            self.data[0, 0] = -1           
            self.data[6, 6] = -1
            self.data[0, 6] = 1
            self.data[6, 0] = 1
        else:
            self.data = board.copy()
            
    def reset(self, board=None):
        if board is None:
            self.data = np.zeros((7, 7), dtype=np.int8)
            self.data[0, 0] = -1           
            self.data[6, 6] = -1
            self.data[0, 6] = 1
            self.data[6, 0] = 1
        else:
            self.data = board.copy()
                
    def is_valid(self, turn, pos, get_pos=False):
        r = pos[0]
        c = pos[1]
        if self.data[r, c] != 0:
            if not get_pos:
                return False
            else:
                return
        else:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    new_r = r+dr
                    new_c = c+dc
                    if new_r >= 0 and new_c >= 0 and new_r < 7 and new_c < 7 and self.data[new_r, new_c] == turn:
                        if not get_pos:
                            return True
                        else:
                            yield new_r, new_c, dr, dc
            if not get_pos:
                return False
        
    def get_moves(self, turn, return_node_info=False):
        action_mask = np.zeros(792, dtype=np.int8)
        next_moves = []
        corr_dict = {}
        children_dict = {}
        for r in range(7):
            for c in range(7):
                has_duplicate_move = False      # move within the radius of one of another friendly piece is called
                for new_r, new_c, dr, dc in self.is_valid(turn, (r, c), True): # duplicate move
                    if new_r >= 0 and new_c >= 0 and new_r < 7 and new_c < 7 and self.data[new_r, new_c] == turn:
                        if abs(dr) <= 1 and abs(dc) <=1:
                            if has_duplicate_move: 
                                cur_move = ((new_r, new_c), (r, c))
                                corr_dict[cur_move] = dup_move
                                # update action mask
                                if return_node_info: 
                                    action_mask[policy_dict[cur_move]] = 1
                            elif self.data[new_r, new_c] == turn:
                                dup_move = ((new_r, new_c), (r, c))
                                next_moves.append(dup_move) 
                                has_duplicate_move = True
                                # preparing children nodes and action mask
                                if return_node_info: 
                                    children_dict[dup_move] = None
                                    action_mask[policy_dict[dup_move]] = 1
                        elif self.data[new_r, new_c] == turn:
                            cur_move = ((new_r, new_c), (r, c))
                            next_moves.append(cur_move) 
                            # preparing children nodes and action mask
                            if return_node_info:
                                children_dict[cur_move] = None
                                action_mask[policy_dict[cur_move]] = 1
                        else:
                            continue
        if return_node_info:
            return next_moves, corr_dict, children_dict, np.array(action_mask)
        else:
            return next_moves
        
    def move_to(self, turn, pos0, pos1):
        x0 = pos0[0]
        y0 = pos0[1]
        x1 = pos1[0]
        y1 = pos1[1]
        
        if not self.is_valid(turn, pos1):
            raise ValueError("This move: " + str((pos0, pos1)) + " of turn: " + str(turn) + " is invalid") 
        elif self.data[x0, y0] != turn:
            raise ValueError("The starting position is not your piece")
        else:
            self.data[x1, y1] = turn
            if abs(x0 - x1) > 1 or abs(y0 - y1) > 1:   # jump move
                self.data[x0, y0] = 0

            for dr in range(-1, 2):                  # infection mode!!!!
                for dc in range(-1, 2):
                    if x1+dr >= 0 and y1+dc >= 0 and x1+dr < 7 and y1+dc < 7:
                        if self.data[x1+dr, y1+dc] == -turn:  # convert any piece of the opponent to 'turn'
                            self.data[x1+dr, y1+dc] = turn
    
    def evaluate(self, turn, this_turn, max_score=1, min_score=0.001):
        turn_no=0
        op_no=0
        for r in range(7):
            for c in range(7):
                if self.data[r, c] == turn:
                    turn_no += 1
                elif self.data[r, c] == -turn:
                    op_no += 1
        if len(self.get_moves(this_turn)) == 0:# if one of them can no longer move, count and end
            if turn_no > op_no:
                return max_score
            else:
                return -max_score
        else:
            value = turn_no - op_no
        return value * min_score
        
        
'''These methods are for Min max with ataxxzero'''


class PolicyValueNetwork():
    def __init__(self):
        self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24))
        K.set_session(self._sess)
        
        self._model = load_model('AtaxxZero.h5')
        print("successfully loaded two models")
           
    def predict(self, feature_map, action_mask):        
        return self._sess.run(self._model.outputs, feed_dict={self._model.inputs[0]: feature_map.reshape(-1, 6, 9, 9), \
                                self._model.inputs[1]: action_mask.reshape(-1, 792), K.learning_phase(): 0})
    
class MinMaxZero():
    def __init__(self):
        self._evaluator = PolicyValueNetwork()
        self._policy_dict = self.get_policy_dict()
        
    @staticmethod
    def get_policy_dict():
        '''Get the relation between policy no. and policy'''
        index=0
        policy_dict = {}
        for r in range(7):
            for c in range(7):
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        new_r = r + dr
                        new_c = c + dc
                        if (dr != 0 or dc != 0) and (new_r < 7 and new_r >= 0) and (new_c < 7 and new_c >= 0):
                            policy_dict[((r, c), (new_r, new_c))] = index
                            index += 1
        return policy_dict

    @staticmethod
    def get_feature_map(board, turn, pre_move):
        out = np.zeros((6, 9, 9), dtype=np.int8)
        # define 1 edge

        # edge
        for j in range(9):
            for k in range(9):
                if j == 0 or j == 8 or k == 0 or k == 8:
                    out[0, j, k] = 1

        # my pieces
        for j in range(9):
            for k in range(9):
                if j > 0 and j < 8 and k > 0 and k < 8:
                    if board[j-1, k-1] == turn:
                        out[1, j, k] = 1

        # op pieces
        for j in range(9):
            for k in range(9):
                if j > 0 and j < 8 and k > 0 and k < 8:
                    if board[j-1, k-1] == -turn:
                        out[2, j, k] = 1

        # last move
        if not pre_move is None:               
            out[3, pre_move[0][0]+1, pre_move[0][1]+1] = 1
            out[4, pre_move[1][0]+1, pre_move[1][1]+1] = 1

        # whose first
        if turn == -1:
            for j in range(9):
                for k in range(9):
                    out[5, j, k] = 1
        return out
    
    def evaluate(self, feature_map, action_mask, turn, target_turn):
        result = self._evaluator.predict(feature_map, action_mask)
        p = result[0][0]
        if turn == target_turn:
            q = result[1][0]
        else:
            q = -result[1][0]
        return p, q

    @staticmethod
    def is_valid(board, turn, pos):
        r = pos[0]
        c = pos[1]
        if board[r, c] != 0:
            return
        else:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    new_r = r+dr
                    new_c = c+dc
                    if new_r >= 0 and new_c >= 0 and new_r < 7 and new_c < 7 and board[new_r, new_c] == turn:
                        yield new_r, new_c, dr, dc
    
    def get_moves(self, board, turn):
        next_moves = []
        dup_moves = []
        action_mask = np.zeros(792, np.int8)
        for r in range(7):
            for c in range(7):
                has_duplicate_move = False      # move within the radius of one of another friendly piece is called
                for new_r, new_c, dr, dc in self.is_valid(board, turn, (r, c)): # duplicate move
                    cur_move = ((new_r, new_c), (r, c))
                    # update action mask
                    action_mask[self._policy_dict[cur_move]] = 1
                    if abs(dr) <= 1 and abs(dc) <=1:
                        if not has_duplicate_move:
                            has_duplicate_move = True
                            next_moves.append(cur_move)
                    else:
                        next_moves.append(cur_move)

        return next_moves, action_mask
    
    @staticmethod
    def move_to(board, turn, pos0, pos1):
        x0 = pos0[0]
        y0 = pos0[1]
        x1 = pos1[0]
        y1 = pos1[1]

        board = board.copy()
        board[x1, y1] = turn
        if abs(x0 - x1) > 1 or abs(y0 - y1) > 1:   # jump move
            board[x0, y0] = 0

        for dr in range(-1, 2):                  # infection mode!!!!
            for dc in range(-1, 2):
                if x1+dr >= 0 and y1+dc >= 0 and x1+dr < 7 and y1+dc < 7:
                    if board[x1+dr, y1+dc] == -turn:  # convert any piece of the opponent to 'turn'
                        board[x1+dr, y1+dc] = turn
        return board

    @staticmethod
    def display_move_prob(move_prob):
        for move, prob in move_prob:
            print(move, prob)
    
    def min_max(self, board, turn, target_turn, depth=3, alpha=-100, beta=100, is_max=True, is_root=True, pre_move=None):
        '''A recursive alpha beta pruning min_max function
        return: board evaluation, chosen move
        NB. for board evaluation, if the searching was pruned, it will return 100 for a minimizer and -100 for a maximizer'''
        if is_root:
            best_moves = []
        else:
            best_move = ((0, 0), (0, 0))

        next_moves, action_mask = self.get_moves(board, turn)
        feature_map = self.get_feature_map(board, turn, pre_move)

        p, q = self.evaluate(feature_map, action_mask, turn, target_turn)

        if depth == 0: # start to do pruning and selecting once the recursion reaches the end
            return q, None
        elif len(next_moves) == 0:
            if (board == target_turn).sum() - (board == -target_turn).sum() > 0:
                return 1, None
            else:
                return -1, None
        else:
            # generate move corresponding p list
            move_prob = []
            all_prob = 0.0
            for move in next_moves:
                prob = p[self._policy_dict[move]]
                move_prob.append((move, prob))
                all_prob += prob
            move_prob = sorted(move_prob, key=lambda x: x[1], reverse=True)

            if is_max:
                alpha = -100
            else:
                beta = 100

            sum_prob = 0.0
            counter = 0
            counter_thresh = len(move_prob) / 15.0
            prob_thresh = (all_prob / len(move_prob)) * 0.8
            # display_move_prob(move_prob)
            for move, prob in move_prob:
                sum_prob += prob
                counter += 1
                # do searching
                try:
                    result, _ = self.min_max(self.move_to(board, turn, move[0], move[1]), \
                                    -turn, target_turn, depth-1, alpha, beta, not is_max, False, move)
                except:
                    print(move)
                    raise
                # prun the searching tree or update alpha and beta respectively
                if is_max:
                    if result >= beta:
                        return 100, None
                    elif result > alpha:
                        alpha = result
                        if is_root:
                            best_moves = [move]
                        else:
                            best_move = move
                    elif result == alpha and is_root:
                        best_moves.append(move)
                else:
                    if result <= alpha:
                        return -100, None
                    elif result < beta:
                        beta = result
                        if is_root:
                            best_moves = [move]
                        else:
                            best_move = move
                    elif result == beta and is_root:
                        best_moves.append(move)

                if sum_prob >= prob_thresh and counter >= counter_thresh:
                    break

            if is_max:
                if is_root:
                    return alpha, choice(best_moves)
                else:
                    return alpha, best_move
            else:
                if is_root:
                    return beta, choice(best_moves)
                else:
                    return beta, best_move
            
            
def recover_game():
    """recover game from input log"""
    game = Ataxx()

    full_input = json.loads(input())
    if "data" in full_input:
        my_data = full_input["data"];
    else:
        my_data = None

    all_requests = full_input["requests"]
    all_responses = full_input["responses"]

    if int(all_requests[0]["x0"]) < 0:
        my_turn = -1
    else:
        my_turn = 1

    for i in range(len(all_responses)):
        pos0, pos1 = (all_requests[i]['x0'], all_requests[i]['y0']), \
    (all_requests[i]['x1'], all_requests[i]['y1'])
        if (pos1[0] >= 0):
            game.move_to(-my_turn, pos0, pos1) 
        pos0, pos1 = (all_responses[i]['x0'], all_responses[i]['y0']), \
    (all_responses[i]['x1'], all_responses[i]['y1'])
        if (pos1[0] >= 0):
            game.move_to(my_turn, pos0, pos1)

    i = len(all_responses)
    pos0, pos1 = (all_requests[i]['x0'], all_requests[i]['y0']), \
    (all_requests[i]['x1'], all_requests[i]['y1'])
    if (pos1[0] >= 0):
        game.move_to(-my_turn, pos0, pos1) 
        pre_move = (pos0, pos1)
    else:
        pre_move = None
        
    return game, my_turn, pre_move

def broadcast_move(move, debug_info=''):
    this_response = {'x0': move[0][0], 'y0': move[0][1], 'x1':move[1][0], 'y1':move[1][1]}
    print(json.dumps({
        "response": this_response,
        "debug": debug_info
    }))

def get_dep_lim(current_game):
    space = (current_game.data == 0).sum()
    if space > 44:
        return 4
    elif space < 5:
        return 6
    elif space < 3:
        return 5
    else:
        return 3

# create a player
minmax_zero = MinMaxZero()
# recover the game board
current_game, my_turn, pre_move = recover_game()

# get move
dep_lim = get_dep_lim(current_game)

C_T = time.time() - round_start

_, move = minmax_zero.min_max(current_game.data, my_turn, my_turn, depth=dep_lim, pre_move=pre_move)

S_T = time.time() - round_start - C_T

# broadcast move
debug_info = "dep_lim {}, configuration time {}, searching time {}".format(dep_lim, C_T, S_T)
broadcast_move(move, debug_info)
