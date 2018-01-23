#coding:utf-8
from __future__ import print_function

from BLEU import compute_bleu
import jieba
import json


def split_sentence(sentence):
    return [i for i in jieba.cut(sentence)]


def load_references_and_inferences(reference_json_path, inference_json_path):
    with open(reference_json_path, "r") as fr:
        references_data = json.load(fr)
    with open(inference_json_path,"r") as fr:
        inference_data = json.load(fr)

    assert len(references_data)==len(inference_data), "the size of references json not equal to inference json"

    references_dict = {}
    for data in references_data:
        image_id = data["image_id"]
        references = data["caption"]
        references_dict[image_id] = references

    references_corpus = []
    inferences_corpus = []
    for data in inference_data:
        image_id = data["image_id"]+".jpg"
        inference = data["caption"]
        inferences_corpus.append(split_sentence(inference))
        references_corpus.append([split_sentence(i) for i in references_dict[image_id]])

    return references_corpus, inferences_corpus


def compute_bleu_score(reference_json_path, inference_json_path, max_order):
    references_corpus, inferences_corpus = load_references_and_inferences(reference_json_path, inference_json_path)
    bleu_score, _, _, _, _, _ = compute_bleu(references_corpus, inferences_corpus, max_order=max_order)
    return bleu_score

print(compute_bleu_score("/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json",
                       "/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_validation_20170910/caption_validation_inference.json",
                       1))