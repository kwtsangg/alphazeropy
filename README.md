# Alphazeropy
This is a python-based platform on which AlphaZero is implemented to train and play board games including Go, Gomoku, ConnectFour and Reversi.

I am glad that my project comes to an end now. I have learnt a lot during creating this python code.

![](connectfour/figs/snap_connectfour.png)

p.s. Press space bar to see what moves are available

# Features
1. python2 and python3 compatible code
1. possible to train the AI using TPU on colab
1. many board games on the same infra-structure
1. simple pygame gui inferface

# Setup
First make sure that the python version is 3.7.16.
Then we can create a virtual environment and download packages.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# I just want to play

Take a look at the 'help'.

```bash
python play.py --help
```

## Reversi

### Example command
```bash
python play.py --game reversi --p1-brain reversi/trained_model/201904290601_reversi_board_8_8_res_blocks_10_filters_48
```
or simply
```
make play_reversi
```

### Training summary
![](reversi/figs/elo_vs_game.png)

I uploaded four AIs which have elo of 2051, 2542, 3083 and 3575.
They are all under reversi/trained_model.
Feel free to play against the current strongest AI which has played 6454 game.

## Connectfour

### Example command
```bash
python play.py --game connectfour --p1-brain connectfour/trained_model/201907252139_connectfour_n_in_row_4_board_6_7_res_blocks_10_filters_48
```
or simply
```
make play_connectfour
```

### Training summary
![](connectfour/figs/elo_vs_game.png)

I uploaded four AIs which have elo of 1675, 2463, 3680 and 4523.
They are all under connectfour/trained_model.
Feel free to play against the current strongest AI which has played 7991 game.

## Others
I am not going to train for other games for now.

# I want to train a model
1. Modify the makefile
1. Create a brain by 'make brain'
1. Open terminals to generate data by running 'make generate'
1. Train the model using latest game play by running 'make train'

# Snapshot

## Game generation
I generate the gameplay with 8 terminals.
![](reversi/figs/snap_selfplaying.png)

## Training on TPU
TPU is awesome! It is at least 10 times faster than GPU!
Because tensorflow-2.0 is not compatible to TPU for now, so 1.13.0rc2 should be used.
![](reversi/figs/snap_finaltraining.png)

# References
1. [AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
1. [AlphaGo Zero: Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
1. [AlphaGo: Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
1. [junxiaosong AlphaZero_Gomoku github](https://github.com/junxiaosong/AlphaZero_Gomoku/)
1. [Bayesian Optimization in AlphaGo](https://arxiv.org/abs/1812.06855)

