#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="1"
python build_tfrecord.py --image_dir=/home/leon/MachineLearning/AIChallenger_caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910\
                        --captions_file=/home/leon/MachineLearning/AIChallenger_caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json\
                        --output_dir=/home/leon/MachineLearning/AIChallenger_caption/ai_challenger_caption_train_output\
                        --train_shards=280\
                        --num_threads=4\
			--word_counts_output_file=/home/leon/MachineLearning/AIChallenger_caption/ai_challenger_caption_train_output/word_counts.txt
