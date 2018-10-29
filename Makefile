game=reversi
n_in_row=4
board_height=8
board_width=8
rollout=800
gpu_memory=0.6

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
 --train-on-last-n-sets 5000 \
 --gpu-memory ${gpu_memory}

online:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter 32 \
 --n-rollout ${rollout} \
 --batch-size 1024 \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --epochs 200 \
 --learning-rate 1e-2 \
 --learning-rate-f 1e-4 \
 --train-online \
 --train-on-last-n-sets 500 \
 --gpu-memory ${gpu_memory}

generate:
	python train.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --board-height ${board_height} \
 --board-width ${board_width} \
 --n-filter 32 \
 --n-rollout ${rollout} \
 --save-path ${PWD}/${game}_training_model/ \
 --load-latest-model \
 --c-puct 5 \
 --generate-game-data-only \
 --gpu-memory ${gpu_memory}

evaluate:
	python play.py \
 --game ${game}\
 --n-in-row ${n_in_row} \
 --evaluate \
 --evaluate-game 300

