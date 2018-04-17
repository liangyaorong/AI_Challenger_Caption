import shutil
import os
import random
import json

# scr_dir = "./ai_challenger_caption_validation_20170910/caption_validation_images_20170910/"
# target_dir = "./ai_challenger_caption_validation_5000/caption_validation_images_5000/"
#
# all_image_list = os.listdir(scr_dir)
# print all_image_list[0]
# random.shuffle(all_image_list)
# print all_image_list[0]
# partial_list = all_image_list[0:5000]
# for image in partial_list:
#     shutil.copyfile(scr_dir+image, target_dir+image)

scr_dir = "./ai_challenger_caption_validation_5000/caption_validation_images_5000"
old_json = "./ai_challenger_caption_validation_5000/caption_validation_annotations_20170910.json"
new_json = "./ai_challenger_caption_validation_5000/caption_validation_annotations_5000.json"

file_name_list = os.listdir(scr_dir)

new_json_data = []

with open(old_json,"r") as fr:
    data = json.load(fr)
for i in data:
    if i["image_id"] in file_name_list:
        new_json_data.append(i)

with open(new_json, "w") as fr:
    json.dump(new_json_data, fr)