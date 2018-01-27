import os
import tensorflow as tf
import itertools
import numpy as np
from random import choice
np.random.seed(1337)  # for reproducibility
from keras.datasets import cifar100
from keras.models import Sequential, Model, load_model
from keras.layers import Input, BatchNormalization, Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AlphaDropout, ConvLSTM2D, AvgPool2D
from keras.layers import add, concatenate
from keras.initializers import VarianceScaling, RandomUniform
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.regularizers import l1, l2
import keras.backend as K
K.set_image_dim_ordering('th')

from keras.callbacks import Callback, ReduceLROnPlateau, LearningRateScheduler, TensorBoard, ModelCheckpoint
import time

class Ataxx:
    def __init__(self, board=None, turn=None):
        if board is None:                  # if there is no initialization given
            self.data = np.zeros((7, 7))   # then generate a board with starting init, and black(-1) takes first turn
            self.data[0, 0] = -1           
            self.data[6, 6] = -1
            self.data[0, 6] = 1
            self.data[6, 0] = 1
        else:
            self.data = board
                
    def is_valid(self, turn, pos):
        if turn not in [-1, 1]:
            raise ValueError("Turn must be -1 or 1") 
        if self.data[pos] != 0:
            return False
        else:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    pos_tmp = (pos[0]+dr, pos[1]+dc)
                    if pos_tmp[0] >= 0 and pos_tmp[1] >= 0 and pos_tmp[0] < 7 and pos_tmp[1] < 7:
                        if self.data[pos_tmp] == turn:  # convert any piece of the opponent to 'turn'
                            return True
            return False
        
    def get_moves(self, turn):
        if turn not in [-1, 1]:
            raise ValueError("Turn must be -1 or 1")# return a list of tuples of tuple
        else:
            next_moves = []
            for r in range(7):
                for c in range(7):
                    has_duplicate_move = False      # move within the radius of one of another friendly piece is called
                    if self.is_valid(turn, (r, c)): # duplicate move
                        for dr in range(-2, 3):
                            for dc in range(-2, 3):
                                if abs(dr) <= 1 and abs(dc) <=1 and has_duplicate_move: 
                                    continue        # no need to record same move again
                                else:
                                    pos_tmp = (r+dr, c+dc)
                                    if pos_tmp[0] >= 0 and pos_tmp[1] >= 0 and pos_tmp[0] < 7 and pos_tmp[1] < 7:
                                        if self.data[pos_tmp] == turn:
                                            next_moves.append((pos_tmp, (r, c)))
            return next_moves
        
    def move_to(self, turn, pos0, pos1):
        if turn not in [-1, 1]:
            raise ValueError("Turn must be -1 or 1") 
        elif not self.is_valid(turn, pos1):
            raise ValueError("This move: " + str(pos1) + " of turn: " + str(turn) + " is invalid") 
        elif self.data[pos0] != turn:
            raise ValueError("The starting position is not your piece")
        else:
            dis = np.array(pos1) - np.array(pos0)    # check if is jump move or duplicate move
            if abs(dis[0]) > 1 or abs(dis[1]) > 1:   # jump move
                self.data[pos0] = 0
                self.data[pos1] = turn
            else:                                    # duplicate move
                self.data[pos1] = turn
            for dr in range(-1, 2):                  # infection mode!!!!
                for dc in range(-1, 2):
                    pos_tmp = (pos1[0]+dr, pos1[1]+dc)
                    if pos_tmp[0] >= 0 and pos_tmp[1] >= 0 and pos_tmp[0] < 7 and pos_tmp[1] < 7:
                        if self.data[pos_tmp] == -turn:  # convert any piece of the opponent to 'turn'
                            self.data[pos_tmp] = turn
    
    def evaluate(self, turn, this_turn):
        if turn not in [-1, 1]:
            raise ValueError("Turn must be -1 or 1") 
        else:
            turn_no = 0
            op_no = 0
            for r in range(7):
                for c in range(7):
                    if self.data[r, c] == turn:
                        turn_no += 1
                    elif self.data[r, c] == -turn:
                        op_no += 1
            if turn_no + op_no == 49:
                if turn_no > op_no:
                    return 1000
                else:
                    return -1000
            else:
                if len(self.get_moves(this_turn)) == 0:# if one of them can no longer move, count and end
                    if turn_no > op_no:
                        return 1000
                    else:
                        return -1000
                else:
                    return turn_no - op_no
                

game = Ataxx()

import json

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

move = choice(game.get_moves(my_turn))
this_response = {'x0': move[0][0], 'y0': move[0][1], 'x1':move[1][0], 'y1':move[1][1]}
print(json.dumps({
    "response": this_response,
}))

