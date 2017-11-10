# coding:utf-8

import tensorflow as tf
import numpy as np
import random
import json
import jieba

from collections import Counter
from collections import namedtuple

import threading

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


ImageMetadata = namedtuple("ImageMetadata", ["filename", "captions"])

class WordDecoder(object):
    def __init__(self, word_id_dict, unknown_word_id):
        self.word_id_dict = word_id_dict
        self.unknown_word_id = unknown_word_id

    def word2id(self, word):
        if word in self.word_id_dict:
            return self.word_id_dict[word]
        else:
            return self.unknown_word_id


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _process_caption_jieba(caption):
    '''
    jieba分词并加上句首句尾
    :param caption: caption of image
    :return: ["<S>", "word0", "word1", "word2",...,"</S>"]
    '''
    tokenized_caption = ["<S>"]
    tokenized_caption.extend(jieba.cut(caption, cut_all=False))
    tokenized_caption.append("</S>")
    return tokenized_caption


def _load_image_caption(caption_path):
    '''
    读取caption
    :param caption_path: caption path
    :return: [ImageMetaData(image_id, all_tokenized_captions)]
    '''
    with open(caption_path, "r") as fr:
        caption_data = json.load(fr)

    image_meta_data = []
    for data in caption_data:
        image_id = data["image_id"]
        descriptions = data["caption"]
        all_tokenized_caption = []
        for description in descriptions:
            temp_caption = description.strip().replace("/n", "")
            tokenized_caption = _process_caption_jieba(temp_caption)
            all_tokenized_caption.append(tokenized_caption)
        image_meta_data.append(ImageMetadata(image_id, all_tokenized_caption))
    return image_meta_data


def _create_word_dict(all_tokenized_captions, min_word_count, word_count_output_file_path=None):
    '''
    根据所有caption建立词库
    :param all_tokenized_captions: 所有分好词的描述 [[words]]
    :param min_word_count: 总出现次数少于该值则统一标记为"其他词"
    :param word_count_output_file_path: 将词典保存到本地的路径
    :return: 词典{word:frequency}
    '''
    counter = Counter()
    for tokenized_caption in all_tokenized_captions:
        counter.update(tokenized_caption)
    counter_after_truncate = [word_count for word_count in counter.items() if word_count[1]>min_word_count]
    counter_after_truncate.sort(key=lambda x:x[1], reverse=True)

    if word_count_output_file_path is not None:
        with tf.gfile.FastGFile(word_count_output_file_path, "w") as f:
            f.write("\n".join(["%s %s"%(w, c) for w, c in counter_after_truncate]))
        print "write vocabulary file in %s" % word_count_output_file_path

    words = [i[0] for i in counter_after_truncate]
    words_ids = dict([(word, id) for id, word in enumerate(words)])
    return words_ids


def _to_sequence_example(image_path, image_meta_data, wordDecoder):
    '''
    将每张图片和其对应的一句描述封装成sequenceExample
    :param image_path: 照片所在文件夹
    :param image_meta_data: ImageMetaData对象,里面有一句caption
    :param wordDecoder: caption2vet
    :return: 将照片和对应caption,caption_id封装在一起的sequence_example
    '''
    image_full_filename = image_path + "/" + image_meta_data.filename
    with tf.gfile.FastGFile(image_full_filename, "r") as f:
        encode_image = f.read()
    context = tf.train.Features(
        feature={
            "image_name": _bytes_feature(image_meta_data.filename),
            "image_data": _bytes_feature(encode_image)
        }

    )
    assert len(image_meta_data.captions) == 1
    caption = image_meta_data.captions[0]
    caption_ids = [wordDecoder.word2id(word=word) for word in caption]
    feature_lists = tf.train.FeatureLists(
        feature_list={
            "image_caption": _bytes_feature_list(caption),
            "image_caption_id": _int64_feature_list(caption_ids)
        }
    )
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

    return sequence_example


def _build_tfRecord_single_thread(images_path, output_path, separate_image_meta_data, image_range, wordDecoder, num_batches, thread_id):
    # 将写到同一batch的data_id放在同一列表里
    data_id_in_same_batch = []
    for i in range(num_batches):
        data_id_in_same_batch.append([])
    for data_id in image_range:
        _in_batch_id = data_id % num_batches
        data_id_in_same_batch[_in_batch_id].append(data_id)

    # 对每个batch
    for batch_id in range(num_batches):
        output_path_detail = output_path + "/" + "image_caption_%s_of_%s" % (batch_id, num_batches)
        writer = tf.python_io.TFRecordWriter(output_path_detail)
        # 对batch中的每个data
        for data_id in data_id_in_same_batch[batch_id]:
            sequence_example = _to_sequence_example(images_path, separate_image_meta_data[data_id], wordDecoder)
            writer.write(sequence_example.SerializeToString())
        print "finish building batch %s in thread %s" % (batch_id+1, thread_id)

        writer.close()

def _build_tfRecord(images_path, separate_image_meta_data, output_path, num_batches, num_threads, wordDecoder):
    num_image = len(separate_image_meta_data)
    shard_range = np.linspace(0, num_image, num_threads+1).astype("int") #将数据分成num_threads份, 如[0,5,10,15]四个数确定三个区间

    # 将确定每个线程要处理的id
    range_list = []
    for i in range(len(shard_range)-1):
        range_list.append(range(shard_range[i], shard_range[i+1]))

    print "start building TFRecords"
    
    coord = tf.train.Coordinator()
    threads = []
    for thread_id in range(num_threads):
        args = (images_path, output_path, separate_image_meta_data, range_list[thread_id], wordDecoder, num_batches, thread_id)
        t = threading.Thread(target=_build_tfRecord_single_thread, args=args)
        t.start()
        threads.append(t)
    coord.join(threads)

    print "finished building all TFRecords"


if __name__ == "__main__":
    images_path = "./ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
    caption_path = "./ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json"
    TFRecord_output_path = "./ai_challenger_caption_TFRecords"

    images_meta_data_list = _load_image_caption(caption_path)

    # 建词典
    all_captions = [caption for image in images_meta_data_list for caption in image.captions]
    word_dict = _create_word_dict(all_captions, min_word_count=4)
    word_decoder = WordDecoder(word_dict, unknown_word_id=5000000)

    # 将每一句caption单独关联照片
    separate_images_meta_data_list = [
        ImageMetadata(image.filename, [caption]) for image in images_meta_data_list for caption in image.captions
    ]

    # 乱序
    random.seed(12345)
    random.shuffle(separate_images_meta_data_list)

    _build_tfRecord(images_path, separate_images_meta_data_list, TFRecord_output_path, 100, 4, word_decoder)



