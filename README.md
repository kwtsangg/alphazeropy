## alphazeropy
This is a python-based platform on which AlphaZero is implemented to train and play board games.

## References
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
1. AlphaGo Zero: Mastering the game of Go without human knowledge
1. AlphaGo: Mastering the game of Go with deep neural networks and tree search
1. [junxiaosong AlphaZero_Gomoku github](https://github.com/junxiaosong/AlphaZero_Gomoku/)

## I just want to play
Example:
python --game connectfour --p1-brain connectfour/trained_model/201809292054_connectfour_n_in_row_4_board_6_7_res_blocks_5_filters_32

## I want to train a model
1. Create a brain by 'make brain'
1. Open terminals to generate data by running 'make generate'
1. Train the model using latest game play by running 'make train'
