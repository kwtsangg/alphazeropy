game=reversi
n_in_row=4
board_height=8
board_width=8
rollout=800
n_filter=48
batch_size=1024
epochs=200

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
 --learning-rate 1e-2 \
 --learning-rate-f 1e-4 \
 --train-on-game-data-only \
 --train-on-last-n-sets 5000

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
 --learning-rate 1e-2 \
 --learning-rate-f 1e-4 \
 --train-online \
 --train-on-last-n-sets 500

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
 --c-puct 5 \
 --generate-game-data-only

evaluate:
	python play.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --evaluate \
 --evaluate-game 300

brain:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter ${n_filter} \
 --save-path ${PWD}/${game}_training_model/ \

