game=reversi
n_in_row=4
board_height=8
board_width=8
rollout=400
n_filter=48
batch_size=1024
epochs=200
n_res_blocks=10
train_engine=gpu
c_puct=5
lr_i=1e-2
lr_f=1e-4
last_n_sets=5000

make:
	make generate

train:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter ${n_filter} \
 --batch-size ${batch_size} \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --epochs ${epochs} \
 --learning-rate ${lr_i} \
 --learning-rate-f ${lr_f} \
 --train-on-game-data-only \
 --train-on-last-n-sets ${last_n_sets} \
 --engine ${train_engine}

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
 --engine cpu

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


