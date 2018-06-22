python train.py \
 --game connectfour \
 --n-in-row 4 \
 --board-height 6 \
 --board-width 7 \
 --n-filter 32 \
 --n-rollout 800 \
 --save-path ${PWD}/connectfour_training_model/ \
 --c-puct 5 \
 --generate-game-data-only
