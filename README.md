## alphazeropy
This is a python-based platform on which AlphaZero is implemented to train and play board games including Go, Gomoku, ConnectFour and Reversi.
However, I have only a GTX960M on my laptop. That's why I didnt train an AI for all board games, esp. Go and Gomoku.

## References
1. [AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
1. [AlphaGo Zero: Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
1. [AlphaGo: Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
1. [junxiaosong AlphaZero_Gomoku github](https://github.com/junxiaosong/AlphaZero_Gomoku/)
1. [Bayesian Optimization in AlphaGo](https://arxiv.org/abs/1812.06855)

## I just want to play
python --game connectfour --p1-brain connectfour/trained_model/201809292054_connectfour_n_in_row_4_board_6_7_res_blocks_5_filters_32

## I want to train a model
1. Create a brain by 'make brain'
1. Open terminals to generate data by running 'make generate'
1. Train the model using latest game play by running 'make train'
