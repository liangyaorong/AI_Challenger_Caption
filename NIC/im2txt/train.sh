#!/usr/bin/env bash
#outpath=/home/leon/MachineLearning/AIChallenger_caption/model/train.log
cd /home/leon/MachineLearning/AIChallenger_caption/im2txt/im2txt
TFRECORD_DIR="/home/leon/MachineLearning/AIChallenger_caption/ai_challenger_caption_train_output"
INCEPTION_CHECKPOINT="/home/leon/MachineLearning/AIChallenger_caption/model/inception_v3.ckpt"
MODEL_DIR="/home/leon/MachineLearning/AIChallenger_caption/model"
export CUDA_VISIBLE_DEVICES="1"
python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-00280.tfrecord" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --number_of_steps=500 #> ${outpath} 2>&1 &