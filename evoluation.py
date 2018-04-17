#coding:utf-8
# 参考描述的分词用“ ”隔开
from __future__ import print_function

from BLEU import compute_bleu
import jieba
import json




def split_sentence_with_jieba(sentence):
    return [i for i in jieba.cut(sentence)]

def split_sentence(sentence, sign):
    return sentence.split(sign)

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
        inferences_corpus.append(split_sentence_with_jieba(inference))
        references_corpus.append([split_sentence_with_jieba(i) for i in references_dict[image_id]])# 抽样测试
        # references_corpus.append([split_sentence(i," ") for i in references_dict[image_id]])# 全量测试

    return references_corpus, inferences_corpus


def compute_bleu_score(reference_json_path, inference_json_path, max_order):
    '''

    :param reference_json_path: 参考答案路径,格式：[{image_id:..., caption:[caption1,caption2,...]},{...}]
    :param inference_json_path: 预测答案路径
    :param max_order: 阶数
    :return: BLEU SCORE
    '''
    references_corpus, inferences_corpus = load_references_and_inferences(reference_json_path, inference_json_path)
    bleu_score, _, _, _, _, _ = compute_bleu(references_corpus, inferences_corpus, max_order=max_order)
    return bleu_score

print(compute_bleu_score(
    "/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_validation_5000/caption_validation_annotations_5000.json",
    "/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_validation_5000/caption_validation_images_5000_attention_1_204396_inference.json",
    # "/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_test_a_20180103/caption_test_a_annotations_20180103_new.json",
    # "/home/leon/ML/AI_Challenger/Image_Caption/ai_challenger_caption_test_a_20180103/caption_test_a_attention_1_197124_inference.json",
    3)
)