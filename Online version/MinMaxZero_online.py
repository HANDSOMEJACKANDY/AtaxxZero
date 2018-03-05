import time
round_start = time.time()

import json
import tensorflow as tf
import numpy as np
from math import sqrt, log, exp
from random import choice, random, sample, seed
from keras.models import load_model
import keras.backend as K
K.set_image_dim_ordering('th')
seed()

class Ataxx:
    def __init__(self, board=None):
        self.data = np.zeros((7, 7), dtype=np.int8)   # then generate a board with starting init, and black(-1) takes first turn
        self.data[0, 0] = -1           
        self.data[6, 6] = -1
        self.data[0, 6] = 1
        self.data[6, 0] = 1
    
    def move_to(self, turn, pos0, pos1):
        x0 = pos0[0]
        y0 = pos0[1]
        x1 = pos1[0]
        y1 = pos1[1]
        
        self.data[x1, y1] = turn
        if abs(x0 - x1) > 1 or abs(y0 - y1) > 1:   # jump move
            self.data[x0, y0] = 0

        for dr in range(-1, 2):                  # infection mode!!!!
            for dc in range(-1, 2):
                if x1+dr >= 0 and y1+dc >= 0 and x1+dr < 7 and y1+dc < 7:
                    if self.data[x1+dr, y1+dc] == -turn:  # convert any piece of the opponent to 'turn'
                        self.data[x1+dr, y1+dc] = turn
                            
'''These methods are for Min max with ataxxzero'''
class PolicyValueNetwork():
    def __init__(self, model_name):
        self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24))
        K.set_session(self._sess)
        
        self._model = load_model(model_name)
           
    def predict(self, feature_map, action_mask):        
        return self._sess.run(self._model.outputs, feed_dict={self._model.inputs[0]: feature_map.reshape(-1, 6, 9, 9), \
                                self._model.inputs[1]: action_mask.reshape(-1, 792), K.learning_phase(): 0})
    
    
class MinMaxZero():
    def __init__(self, p_thresh, c_thresh, model_name="/data/AtaxxZero.h5"):
        self._evaluator = PolicyValueNetwork(model_name)
        self._policy_dict = self.get_policy_dict()
        self._p_thresh = p_thresh
        self._c_thresh = c_thresh
        
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
        out[0, 0, :] = 1
        out[0, 8, :] = 1
        out[0, :, 0] = 1
        out[0, :, 8] = 1

        # my pieces
        out[1, 1:8, 1:8] = (board == turn)

        # op pieces
        out[2, 1:8, 1:8] = (board == -turn)
        
        # last move
        if not pre_move is None:               
            out[3, pre_move[0][0]+1, pre_move[0][1]+1] = 1
            out[4, pre_move[1][0]+1, pre_move[1][1]+1] = 1

        # whose first
        if turn == -1:
            out[5, ...] = 1
        return out
    
    def evaluate(self, feature_map, action_mask, turn, target_turn):
        result = self._evaluator.predict(feature_map, action_mask)
        p = result[0][0]
        if turn == target_turn:
            q = result[1][0][0]
        else:
            q = -result[1][0][0]
        return p, q

    @staticmethod
    def is_valid(board, turn, pos):
        r = pos[0]
        c = pos[1]
        if board[r, c] != 0:
            return
        else:
            turn_board = board == turn
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    new_r = r+dr
                    new_c = c+dc
                    try:
                        assert new_r >= 0 and new_c >= 0 and turn_board[new_r, new_c]
                        yield new_r, new_c, dr, dc
                    except:
                        pass
    
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
    
    def min_max(self, board, turn, target_turn, depth=3, alpha=-100, beta=100, is_max=True, is_root=True, \
                pre_move=None, t_lim=1):
        '''A recursive alpha beta pruning min_max function
        return: board evaluation, chosen move
        NB. for board evaluation, if the searching was pruned, it will return 100 for a minimizer and -100 for a maximizer'''
        if is_root:
            best_moves = []
            self._start = time.time()
        else:
            best_move = ((0, 0), (0, 0))
        
        # get next moves
        next_moves, action_mask = self.get_moves(board, turn)
        # stop searching if the game is over
        if len(next_moves) == 0:
            diff = (board == target_turn).sum() - (board == -target_turn).sum()
            if diff > 0:
                return 1, None
            elif diff < 0:
                return -1, None
            else:
                turn_no = (board == turn).sum()
                if turn == -target_turn:
                    turn_no = 49 - turn_no # set turn_no to represent the number of target turn pieces
                if turn_no >= 45:
                    return 1, None
                else:
                    return -1, None
        else: # otherwise calculate p and q and do the NN pruned minmax searching
            feature_map = self.get_feature_map(board, turn, pre_move)
            p, q = self.evaluate(feature_map, action_mask, turn, target_turn)
            
        if depth == 0: # once the recursion reaches the end, return the leaf node value
            return q, None
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
            counter_thresh = len(move_prob) / self._c_thresh
            prob_thresh = all_prob * self._p_thresh
            # display_move_prob(move_prob)
            for move, prob in move_prob:
                sum_prob += prob
                counter += 1
                # do searching
                result, _ = self.min_max(self.move_to(board, turn, move[0], move[1]), \
                                    -turn, target_turn, depth-1, alpha, beta, not is_max, False, move, t_lim)
                # record all the move and its alpha value
                if is_root:
                    try:
                        move_value.append((move, result))
                    except:
                        move_value = [(move, result)]
                # prun the searching tree or update alpha and beta respectively
                if is_max:
                    if result >= beta:
                        return 100, None
                    elif result > alpha:
                        alpha = result
                        best_move = move
                else:
                    if result <= alpha:
                        return -100, None
                    elif result < beta:
                        beta = result
                        best_move = move

                if (sum_prob >= prob_thresh and counter >= counter_thresh) or time.time() - self._start >= t_lim:
                    break
                    
            if is_root: # incorporate ramdom characteristic
                move_value = sorted(move_value, key=lambda x: x[1], reverse=True)
                max_value = alpha
                for move, value in move_value:
                    if value >= max_value - 0.1 * abs(max_value):
                        try:
                            best_move.append(move)
                        except:
                            best_move = [move]
                    else:
                        break
                best_move = choice(best_move)

            if is_max:
                return alpha, best_move
            else:
                return beta, best_move
 
            
def recover_game():
    """recover game from input log"""
    game = Ataxx()

    full_input = json.loads(input())
    if "data" in full_input:
        my_data = full_input["data"]
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
        
    return game, my_turn, pre_move, my_data

def broadcast(move, debug_info=None, data=None):
    this_response = {'x0': move[0][0], 'y0': move[0][1], 'x1':move[1][0], 'y1':move[1][1]}
    print(json.dumps({
        "response": this_response,
        "debug": debug_info,
        "data": data
    }))

def get_dep_lim(current_game, dep_lim):
    space = (current_game.data == 0).sum()
    if space > 42:
        return 4
    elif space == 41 or space == 42:
        return 3
    elif space == 3 or space == 4:
        return 5
    else:
        return dep_lim

# recover the game board
s = time.time()
current_game, my_turn, pre_move, data = recover_game()
R_T = time.time() - s

# process passed data
if not data is None:
    dep_lim = data[0]
    if data[1] >= 2:
        dep_lim += 1
        data[1] = 0
    elif data[1] <= -1.5:
        dep_lim -= 1
        data[1] = 0
else:
    dep_lim = None

# configurate actor
minmax_zero = MinMaxZero(0.99, 100, "/data/AtaxxZero_91.2p_82.4q.h5")

# get dep lim
dep_lim = get_dep_lim(current_game, dep_lim)

# search for moves
C_T = time.time() - round_start

_, move = minmax_zero.min_max(current_game.data, my_turn, my_turn, depth=dep_lim, pre_move=pre_move, t_lim=5.75 - C_T)

S_T = time.time() - round_start - C_T

# pass data
if data is None:
    data = [None, 0]
if S_T / (5.75 - C_T) > 1:
    data[1] -= 1
elif S_T / (5.75 - C_T) < 0.20:
    data[1] += 1
elif data[1] > 0:
    data[1] = 0
elif data[1] < 0:
    data[1] += 0.5
data[0] = dep_lim
    
# broadcast move
L_T = 6 - (time.time() - round_start)
debug_info = "dep_lim {}, recover time {}, total configuration time {}, \
                searching time {}, left time {}, data {}".format(dep_lim, R_T, C_T, S_T, L_T, data)
broadcast(move, debug_info, data)
