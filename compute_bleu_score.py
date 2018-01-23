#coding:utf-8
from __future__ import print_function
import nltk
import jieba
import json

def split_sentence(sentence):
    return [i for i in jieba.cut(sentence)]


def compute_bleu(references, inference, max_order):
    references_after_split = [split_sentence(i) for i in references]
    inference_after_split = split_sentence(inference)
    return nltk.translate.bleu(references_after_split, inference_after_split, [1.0/max_order]*max_order)


def compute_avg_bleu(reference_json_path, inference_json_path, max_order):
    with open(reference_json_path, "r") as fr:
        references_data = json.load(fr)
    with open(inference_json_path,"r") as fr:
        inference_data = json.load(fr)

    assert len(references_data)==len(inference_data), "the size of references json not equal to inference json"
    image_num = len(references_data)

    references_dict = {}
    for data in references_data:
        image_id = data["image_id"]
        # print(image_id)
        references = data["caption"]
        references_dict[image_id] = references
    bleu_all = [0] * image_num
    for (i, data) in enumerate(inference_data):
        image_id = data["image_id"]
        inference = data["caption"]
        bleu = compute_bleu(references_dict[image_id+".jpg"], inference, max_order)
        bleu_all[i] = bleu
    return sum(bleu_all) / image_num


print(compute_avg_bleu("/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json",
                       "/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_validation_20170910/caption_validation_inference.json",
                       1))
