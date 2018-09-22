game=connectfour
n_in_row=4
board_height=6
board_width=7

make:
	make generate

train:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter 32 \
 --batch-size 1024 \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --epochs 200 \
 --learning-rate 1e-2 \
 --learning-rate-f 1e-4 \
 --train-on-game-data-only \
 --train-on-last-n-sets 10000

generate:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter 32 \
 --n-rollout 800 \
 --save-path ${PWD}/${game}_training_model/ \
 --c-puct 5 \
 --generate-game-data-only

evaluate:
	python play.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --evaluate \
 --evaluate-game 300 \
 --p2-brain ${PWD}/${game}_training_model/201805041130_connectfour_n_in_row_4_board_6_7_res_blocks_5_filters_32 \
 --p1-brain ${PWD}/${game}_training_model/201806220927_connectfour_n_in_row_4_board_6_7_res_blocks_5_filters_32

