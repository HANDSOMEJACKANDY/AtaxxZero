
# AtaxxZero
This algorithm tried to reimplement AlphaGo Zero for Ataxx
however, the computation to train an AI from scratch can be too heavy, given my skills of code optimization and hardware limitation
therefore, minor adjustments are made to the algorithms to make it plausible for this algorithm to give a rather satisfactory result in an acceptable period.

## adaptions and modifications:
1. One major difference between AlphaGo Zero and Ataxx Zero is that Ataxx Zero relies one engineered value function. From the beginning of the training, the q value of each node is a combination of q from the hybrid network and a greedy function (output is monotone increasing with regard to difference of piece no. of each player).
2. Another major modification is Ataxx Zero apply MCTS to a very shallow depth, currently being 3. This change significantly reduce the searching time, thus accelerate training greatly.
3. The combination of 3 layer MCTS and an engineered value function guarantees a good performance of the algorithm in even before training, i.e. hybrid network output random probability and value. The behavior of Ataxx Zero before training should resemble an impaired MinMax Searching with a depth of 3. From a practical perspective, it wins 90% of the game with a greedy player(which attempts to maximize no.my_piece - no.opponent's_piece). With reinforcement learning, the algorithm is expected to behave better.
4. When actually applied in game, I plan to reduce the searching depth to 2 to further improve the speed, but expect the algorithm to work better than itself before training.

## introduction to the repository
1. To access the code that does the training of AtaxxZero, access file ./AtaxxZero/MCTS_1d_policy_both_p_q_with_manual_q.ipynb
2. To access python code uploaded to www.botzone.org, access file ./online_Ataxx.py
3. To access code that experiments many temporal difference algorithms, access files under ./TD

## comments
1. Many Temporal Difference algorithms are experimented before I settled on AlphaGoZero. Basically, TD algorithms can barely play the game, as most of them, after weeks of training, still dont seem to understand the most fundamental rules of Ataxx. This, however, can be viewed as a support for the argument that AlphaGo's success is not the success of Neural Network. Basically, without the help of MCTS, which is a super traditional algorithm for board games, AlphaGoZero can achieve nothing at all. From my point of view, a neural network is nothing but a strong approximater. The success of AlphaGoZero is essentially finding a way to combine a traditional logical algorithms with neural network.
2. The whole project was badly commented, especially for the last period of time when I actually sorted out the outline of the algorithm and started the tedious yet enjoyable two months of iterations of training + improving the algorithm. I really regret that as the most shining and innovative ideas were generated during this process but are all lost with time. When I look back on the project today and start to notice what I huge stuff I have done, I am literally buried with regretfulness.....
