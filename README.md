
# AtaxxZero
This algorithm tried to reimplement AlphaGo Zero for Ataxx
however, the computation to train an AI from scratch can be too heavy, given my skills of code optimization and hardware limitation
therefore, minor adjustments are made to the algorithms to make it plausible for this algorithm to give a rather satisfactory result in an acceptable period.
## adaptions and modifications:
1. One major difference between AlphaGo Zero and Ataxx Zero is that Ataxx Zero relies one engineered value function. From the beginning of the training, the q value of each node is a combination of q from the hybrid network and a greedy function (output is monotone increasing with regard to difference of piece no. of each player).
2. Another major modification is Ataxx Zero apply MCTS to a very shallow depth, currently being 3. This change significantly reduce the searching time, thus accelerate training greatly.
3. The combination of 3 layer MCTS and an engineered value function guarantees a good performance of the algorithm in even before training, i.e. hybrid network output random probability and value. The behavior of Ataxx Zero before training should resemble an impaired MinMax Searching with a depth of 3. From a practical perspective, it wins 90% of the game with a greedy player(which attempts to maximize no.my_piece - no.opponent's_piece). With reinforcement learning, the algorithm is expected to behave better.
4. When actually applied in game, I plan to reduce the searching depth to 2 to further improve the speed, but expect the algorithm to work better than itself before training.
