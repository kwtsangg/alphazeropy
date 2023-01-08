game=connectfour
n_in_row=4
board_height=6
board_width=7
rollout=800
n_filter=48
batch_size=1024
epochs=200
n_res_blocks=10
train_engine=gpu
c_puct=5
lr_i=1e-2
lr_f=1e-4
last_n_sets=10000

make:
	make generate

train:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --batch-size ${batch_size} \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --epochs ${epochs} \
 --learning-rate ${lr_i} \
 --learning-rate-f ${lr_f} \
 --train-on-game-data-only \
 --train-on-last-n-sets ${last_n_sets} \
 --engine ${train_engine} \
 --train-rounds 1

online:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter ${n_filter} \
 --n-rollout ${rollout} \
 --batch-size ${batch_size} \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --epochs ${epochs} \
 --learning-rate ${lr_i} \
 --learning-rate-f ${lr_f} \
 --train-online \
 --train-on-last-n-sets ${last_n_sets}

generate:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter ${n_filter} \
 --n-rollout ${rollout} \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --c-puct ${c_puct} \
 --generate-game-data-only \
 --engine cpu \
 --max-game-gen 1000

evaluate:
	python play.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --evaluate \
 --evaluate-game 300 \
 --engine cpu

brain:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter ${n_filter} \
 --save-path ${PWD}/${game}_training_model/ \
 --n-res-blocks ${n_res_blocks} \
 --engine cpu

play_reversi:
	python play.py --game reversi --p1-brain reversi/trained_model/201904290601_reversi_board_8_8_res_blocks_10_filters_48

play_connectfour:
	python play.py --game connectfour --p1-brain connectfour/trained_model/201907252139_connectfour_n_in_row_4_board_6_7_res_blocks_10_filters_48
