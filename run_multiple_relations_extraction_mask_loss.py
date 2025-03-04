# coding=utf-8
# @Author:yuanxiao and Google AI Language Team Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import sys
from bert import modeling
from bert import optimization
from bert import tokenization
from bert import tf_metrics
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_interactive_predict", False,
    "Whether to run the model in inference mode on user input.")

flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model in as SavedModel.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_token, token_label=None, predicate_value_list=None, predicate_location_list=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_token = text_token
        self.token_label = token_label
        self.predicate_value_list = predicate_value_list
        self.predicate_location_list = predicate_location_list


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 token_label_ids,
                 predicate_matrix_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_label_ids = token_label_ids
        self.predicate_matrix_ids = predicate_matrix_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SKE_2019_Subject_Relation_Object_extraction_Processor(DataProcessor):
    """Processor for the SKE_2019 data set"""

    # SKE_2019 data from http://lic2019.ccf.org.cn/kg

    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), "r", encoding='utf-8') as token_in_f,\
            open(os.path.join(data_dir, "labeling_out.txt"), "r", encoding='utf-8') as labeling_out_f, \
            open(os.path.join(data_dir, "predicate_value_out.txt"), "r", encoding='utf-8') as predicate_value_out_f,\
            open(os.path.join(data_dir, "predicate_location_out.txt"), "r", encoding='utf-8') as predicate_location_out_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f]
            token_label_out_list = [seq.replace("\n", '') for seq in labeling_out_f]
            predicate_value_out_list = [eval(seq.replace("\n", '')) for seq in
                                        predicate_value_out_f]
            predicate_location_out_list = [eval(seq.replace("\n", '')) for seq in predicate_location_out_f]
        examples = list(zip(token_in_list, token_label_out_list, predicate_value_out_list,
                            predicate_location_out_list))
        return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")

    def _raw_token_labels(self):
        return ['Date', 'Number', 'Text', '书籍', '人物', '企业', '作品', '出版社', '历史人物', '国家', '图书作品', '地点', '城市', '学校', '学科专业',
                '影视作品', '景点', '机构', '歌曲', '气候', '生物', '电视综艺', '目', '网站', '网络小说', '行政区', '语言', '音乐专辑']

    def get_token_labels(self):
        raw_token_labels = self._raw_token_labels()
        BIO_token_labels = ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]"]  # id 0 --> [Paddding]
        for label in raw_token_labels:
            BIO_token_labels.append("B-" + label)
            BIO_token_labels.append("I-" + label)
        BIO_token_labels.append("O")
        return BIO_token_labels

    def get_predicate_labels(self):
        "N --> no predicate"
        return ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
                '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
                '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
                '连载网站', '邮政编码', '面积', '首都']

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_token = line
                token_label = None
                predicate_value_list = None
                predicate_location_list = None
            else:
                text_token = line[0]
                token_label = line[1]
                predicate_value_list = line[2]
                predicate_location_list = line[3]
            examples.append(
                InputExample(guid=guid, text_token=text_token, token_label=token_label,
                             predicate_value_list=predicate_value_list,
                             predicate_location_list=predicate_location_list))
        return examples


class SKE_2019_Subject_Relation_Object_extraction_Processor_V2(SKE_2019_Subject_Relation_Object_extraction_Processor):
    """Remove so called predicate `N`"""

    def get_predicate_labels(self):
        return ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
                '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
                '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
                '连载网站', '邮政编码', '面积', '首都']

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), "r", encoding='utf-8') as token_in_f,\
            open(os.path.join(data_dir, "labeling_out.txt"), "r", encoding='utf-8') as labeling_out_f, \
            open(os.path.join(data_dir, "predicate_value_out.txt"), "r", encoding='utf-8') as predicate_value_out_f,\
            open(os.path.join(data_dir, "predicate_location_out.txt"), "r", encoding='utf-8') as predicate_location_out_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f]
            token_label_out_list = [seq.replace("\n", '') for seq in labeling_out_f]
            predicate_value_out_list = [eval(seq.replace("\n", '')) for seq in
                                        predicate_value_out_f]
            predicate_location_out_list = [eval(seq.replace("\n", '')) for seq in predicate_location_out_f]
        assert len(predicate_value_out_list) == len(predicate_location_out_list)
        "Remove the `N` predicate in place"
        predicate_to_keep = set(self.get_predicate_labels())
        for predicate_values, predicate_locations in zip(predicate_value_out_list, predicate_location_out_list):
            assert len(predicate_values) == len(predicate_locations)
            for token_i, (token_i_predicate_values, token_i_predicate_locations) in enumerate(zip(predicate_values, predicate_locations)):
                assert len(token_i_predicate_values) == len(token_i_predicate_locations)
                token_i_predicate_values_new, token_i_predicate_locations_new = [], []
                for kept_predicate_value, kept_predicate_location in filter(lambda pair: pair[0] in predicate_to_keep, zip(token_i_predicate_values, token_i_predicate_locations)):
                    token_i_predicate_values_new.append(kept_predicate_value)
                    token_i_predicate_locations_new.append(kept_predicate_location)
                predicate_values[token_i] = token_i_predicate_values_new
                predicate_locations[token_i] = token_i_predicate_locations_new
        examples = list(zip(token_in_list, token_label_out_list, predicate_value_out_list,
                            predicate_location_out_list))
        return examples


def convert_single_example(ex_index, example, token_label_list, predicate_label_list, max_seq_length,
                           tokenizer, num_predicate_label):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            token_label_ids=[0] * max_seq_length,
            predicate_matrix_ids=((), ()),
            is_real_example=False)

    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    predicate_label_map = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_map[label] = i

    text_token = example.text_token.split(" ")
    if example.token_label is not None:
        token_label = example.token_label.split(" ")
    else:
        token_label = ["O"] * len(text_token)
    assert len(text_token) == len(token_label)

    if len(text_token) > (max_seq_length - 2):  # one for [CLS] and one for [SEP]
        text_token = text_token[0:max_seq_length - 2]
        token_label = token_label[0:max_seq_length - 2]

    if example.predicate_value_list is not None:
        predicate_value_list = example.predicate_value_list
        predicate_location_list = example.predicate_location_list
        assert len(predicate_value_list) == len(predicate_location_list)
    else:
        predicate_value_list = [[] for _ in range(len(token_label))]
        predicate_location_list = [[] for _ in range(len(token_label))]

    predicate_idx_value_pairs = _get_sparse_predicate_matrix(predicate_label_map, predicate_value_list,
                                                             predicate_location_list, max_seq_length)
    # Get sparse representation of target predicate selection matrix
    indices, values = [], []
    for (loc_i, loc_j, label_id), val in predicate_idx_value_pairs:
        # Shift loc by one given the leading `[CLS]` token
        loc_i, loc_j = loc_i + 1, loc_j + 1
        # Position `0` for `[CLS]`, Position `max_seq_length - 1` for `[SEP]`
        if (0 < loc_i < max_seq_length - 1) and (0 < loc_j < max_seq_length - 1):
            indices.append((loc_i, loc_j, label_id))
            values.append(val)
    assert len(indices) == len(values)
    predicate_matrix_ids = indices, values
    tokens = []
    token_label_ids = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[CLS]"])

    for token, label in zip(text_token, token_label):
        tokens.append(token)
        segment_ids.append(0)
        token_label_ids.append(token_label_map[label])

    tokens.append("[SEP]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        token_label_ids.append(0)
        tokens.append("[Padding]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("token_label_ids: %s" % " ".join([str(x) for x in token_label_ids]))
        tf.logging.info("predicate_value_list: %s" % " ".join([str(x) for x in predicate_value_list]))
        tf.logging.info("predicate_location_list: %s" % " ".join([str(x) for x in predicate_location_list]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        token_label_ids=token_label_ids,
        predicate_matrix_ids=predicate_matrix_ids,
        is_real_example=True)
    return feature


def _get_multiple_predicate_matrix(predicate_label_map, predicate_value_list, predicate_location_list, max_seq_length):
    num_label = len(predicate_label_map)
    predicate_matrix = np.zeros((max_seq_length, max_seq_length, num_label), dtype=np.float32)
    for location_i, (predicate_values, predicate_location) in enumerate(zip(predicate_value_list, predicate_location_list)):
        predicate_value_ids = list(map(lambda x: predicate_label_map[x], predicate_values))
        assert len(predicate_value_ids) == len(predicate_location)
        # keep value_ids sorted as required by sparse matrix indices
        for (value_id, location_j) in sorted(zip(predicate_value_ids, predicate_location), key=lambda pair: pair[0]):
            if location_i < max_seq_length and location_j < max_seq_length:
                predicate_matrix[location_i, location_j, value_id] = 1.
            else:
                pass
    return predicate_matrix


def _get_sparse_predicate_matrix(predicate_label_map, predicate_value_list, predicate_location_list, max_seq_length):
    idx_val_pairs = []
    unique_idx_val_pairs = set()
    for location_i, (predicate_values, predicate_location) in enumerate(zip(predicate_value_list, predicate_location_list)):
        for (value, location_j) in zip(predicate_values, predicate_location):
            value_id = predicate_label_map[value]
            if location_i < max_seq_length and location_j < max_seq_length:
                idx_val_pair = ((location_i, location_j, value_id), 1.)
                if idx_val_pair not in unique_idx_val_pairs:
                    idx_val_pairs.append(idx_val_pair)
                    unique_idx_val_pairs.add(idx_val_pair)
            else:
                pass
    return idx_val_pairs


def file_based_convert_examples_to_features(
        examples, token_label_list, predicate_label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, token_label_list, predicate_label_list,
                                         max_seq_length, tokenizer, len(predicate_label_list))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["token_label_ids"] = create_int_feature(feature.token_label_ids)
        predicate_matrix_indices_flat = np.array(feature.predicate_matrix_ids[0]).reshape([-1])
        predicate_matrix_values_flat = np.array(feature.predicate_matrix_ids[1]).reshape([-1])

        features["sparse_predicate_matrix_indices"] = create_int_feature(predicate_matrix_indices_flat)
        features["sparse_predicate_matrix_values"] = create_float_feature(predicate_matrix_values_flat)

        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        context = tf.train.Features(feature=features)
        tf_example = tf.train.SequenceExample(context=context)
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, num_predicate_label, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "sparse_predicate_matrix_indices": tf.io.FixedLenSequenceFeature([3], tf.int64, allow_missing=True),
        "sparse_predicate_matrix_values": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_sparse_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # Sparse to dense
        if "predicate_matrix" not in example:
            indices = example["sparse_predicate_matrix_indices"]
            values = example["sparse_predicate_matrix_values"]
            example["predicate_matrix"] = tf.sparse.to_dense(tf.sparse.reorder(
                tf.SparseTensor(indices=indices, values=values, dense_shape=[seq_length, seq_length, num_predicate_label])),
                default_value=0.
            )
            # Delete things that can't be batched
            del example["sparse_predicate_matrix_indices"], example["sparse_predicate_matrix_values"]
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32, "ToInt32")
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_sparse_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def interactive_input_fn_builder(input_stream, seq_length, num_predicate_label):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def _decode_sparse_record(example):
        """Decodes a record to a TensorFlow example."""
        # Sparse to dense
        if "predicate_matrix" not in example:
            indices = example["sparse_predicate_matrix_indices"]
            values = example["sparse_predicate_matrix_values"]
            example["predicate_matrix"] = tf.sparse.to_dense(tf.sparse.reorder(
                tf.SparseTensor(indices=indices, values=values, dense_shape=[seq_length, seq_length, num_predicate_label])),
                default_value=0.
            )
            # Delete things that can't be batched
            del example["sparse_predicate_matrix_indices"], example["sparse_predicate_matrix_values"]
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32, "ToInt32")
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # Dataset.from_generator() uses tf.py_func, which is not TPU compatible.
        d = tf.data.Dataset.from_generator(input_stream,
                                           {
                                               "input_ids": tf.int64,
                                               "input_mask": tf.int64,
                                               "segment_ids": tf.int64,
                                               "token_label_ids": tf.int64,
                                               "sparse_predicate_matrix_indices": tf.int64,
                                               "sparse_predicate_matrix_values": tf.float32,
                                               "is_real_example": tf.int64

                                           },
                                           output_shapes={
                                               "input_ids": tf.TensorShape([seq_length]),
                                               "input_mask": tf.TensorShape([seq_length]),
                                               "segment_ids": tf.TensorShape([seq_length]),
                                               "token_label_ids": tf.TensorShape([seq_length]),
                                               "sparse_predicate_matrix_indices": tf.TensorShape([None, 3]),
                                               "sparse_predicate_matrix_values": tf.TensorShape([None]),
                                               "is_real_example": tf.TensorShape([])
                                           },
                                           args=None)
        d = d.map(_decode_sparse_record)
        d = d.batch(batch_size=1, drop_remainder=False)
        return d

    return input_fn


def get_spans(loc, labels, tokens):
    assert len(tokens) == len(labels)
    l, r = loc, loc + 1
    if labels[l][0] != "B":
        return None
    type_ = labels[l][2:]
    while r < len(tokens):
        current_token, current_label = tokens[r], labels[r]
        if not (current_token[:2] == "##" or ((current_label[0] == "I") and (current_label[2:] == type_))):
            break
        r += 1
    entity_tokens = tokens[l:r]
    return {"tokens": entity_tokens, "type": type_}


def getHeadSelectionScores(encode_input, hidden_size_n1, label_number):
    """
    Check out this paper https://arxiv.org/abs/1804.07847
    """

    def broadcasting(left, right):
        left = tf.transpose(left, perm=[1, 0, 2])
        left = tf.expand_dims(left, 3)  # [L, B, D, 1]
        right = tf.transpose(right, perm=[0, 2, 1])
        right = tf.expand_dims(right, 0)  # [1, B, D, L]
        B = left + right
        B = tf.transpose(B, perm=[1, 0, 3, 2])  # [B, L, L, D]
        return B

    encode_input_hidden_size = encode_input.shape[-1].value
    u_a = tf.get_variable("u_a", [encode_input_hidden_size, hidden_size_n1])
    w_a = tf.get_variable("w_a", [encode_input_hidden_size, hidden_size_n1])
    v = tf.get_variable("v", [hidden_size_n1, label_number])
    b_s = tf.get_variable("b_s", [hidden_size_n1])

    left = tf.einsum('aij,jk->aik', encode_input, u_a)
    right = tf.einsum('aij,jk->aik', encode_input, w_a)
    outer_sum = broadcasting(left, right)
    outer_sum_bias = outer_sum + b_s
    output = tf.tanh(outer_sum_bias)
    g = tf.einsum('aijk,kp->aijp', output, v)
    return g


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 token_label_ids, predicate_matrix, num_token_labels, num_predicate_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. float Tensor of shape [batch_size, hidden_size]
    # model_pooled_output = model.get_pooled_output()

    #     """Gets final hidden layer of encoder.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    #       to the final hidden of the transformer encoder.
    #     """
    sequence_bert_encode_output = model.get_sequence_output()
    if is_training:
        sequence_bert_encode_output = tf.nn.dropout(sequence_bert_encode_output, keep_prob=0.9)

    with tf.variable_scope("predicate_head_select_loss"):
        bert_sequence_length = sequence_bert_encode_output.shape[-2]
        # shape [batch_size, sequence_length, sequence_length, predicate_label_numbers]
        predicate_score_matrix = getHeadSelectionScores(encode_input=sequence_bert_encode_output, hidden_size_n1=100,
                                                        label_number=num_predicate_labels)
        predicate_head_probabilities = tf.nn.sigmoid(predicate_score_matrix)

        # predicate_head_prediction = tf.argmax(predicate_head_probabilities, axis=3)
        predicate_head_predictions_round = tf.round(predicate_head_probabilities)
        predicate_head_predictions = tf.cast(predicate_head_predictions_round, tf.int32)
        # [batch_size, sequence_length, sequence_length, num_predicate_label]
        predicate_matrix = tf.reshape(predicate_matrix, [-1, bert_sequence_length, bert_sequence_length, num_predicate_labels])
        predicate_sigmoid_cross_entropy_with_logits = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=predicate_score_matrix,
            labels=predicate_matrix)

        def batch_sequence_matrix_max_sequence_length(batch_sequence_matrix):
            """Get the longest effective length of the input sequence (excluding padding)"""
            mask = tf.math.logical_not(tf.math.equal(batch_sequence_matrix, 0))
            mask = tf.cast(mask, tf.float32)
            mask_length = tf.reduce_sum(mask, axis=1)
            mask_length = tf.cast(mask_length, tf.int32)
            mask_max_length = tf.reduce_max(mask_length)
            return mask_max_length

        mask_max_length = batch_sequence_matrix_max_sequence_length(token_label_ids)

        predicate_sigmoid_cross_entropy_with_logits = predicate_sigmoid_cross_entropy_with_logits[
                                                      :, :mask_max_length, :mask_max_length, :]
        # shape []
        predicate_head_select_loss = tf.reduce_sum(predicate_sigmoid_cross_entropy_with_logits)

    with tf.variable_scope("token_label_loss"):
        bert_encode_hidden_size = sequence_bert_encode_output.shape[-1].value
        token_label_output_weight = tf.get_variable(
            "token_label_output_weights", [num_token_labels, bert_encode_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        token_label_output_bias = tf.get_variable(
            "token_label_output_bias", [num_token_labels], initializer=tf.zeros_initializer()
        )
        sequence_bert_encode_output = tf.reshape(sequence_bert_encode_output, [-1, bert_encode_hidden_size])
        token_label_logits = tf.matmul(sequence_bert_encode_output, token_label_output_weight, transpose_b=True)
        token_label_logits = tf.nn.bias_add(token_label_logits, token_label_output_bias)

        token_label_logits = tf.reshape(token_label_logits, [-1, FLAGS.max_seq_length, num_token_labels])
        token_label_log_probs = tf.nn.log_softmax(token_label_logits, axis=-1)

        token_label_one_hot_labels = tf.one_hot(token_label_ids, depth=num_token_labels, dtype=tf.float32)
        token_label_per_example_loss = -tf.reduce_sum(token_label_one_hot_labels * token_label_log_probs, axis=-1)
        token_label_loss = tf.reduce_sum(token_label_per_example_loss)
        token_label_probabilities = tf.nn.softmax(token_label_logits, axis=-1)
        token_label_predictions = tf.argmax(token_label_probabilities, axis=-1)
        # return (token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predict)

    loss = predicate_head_select_loss + token_label_loss
    return (loss,
            predicate_head_select_loss, predicate_head_probabilities, predicate_head_predictions,
            token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions)


def model_fn_builder(bert_config, num_token_labels, num_predicate_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        token_label_ids = features["token_label_ids"]
        predicate_matrix = features["predicate_matrix"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(token_label_ids), dtype=tf.float32)  # TO DO

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,
         predicate_head_select_loss, predicate_head_probabilities, predicate_head_predictions,
         token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            token_label_ids, predicate_matrix, num_token_labels, num_predicate_labels,
            use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(true_predicate_matrix, pred_predicate_matrix,
                          predicate_head_select_loss, token_label_per_example_loss, token_label_ids, token_label_logits,
                          is_real_example):
                token_label_predictions = tf.argmax(token_label_logits, axis=-1, output_type=tf.int32)
                token_label_pos_indices_list = list(range(num_token_labels))[
                                               4:]  # ["[Padding]","[##WordPiece]", "[CLS]", "[SEP]"] + seq_out_set
                pos_indices_list = token_label_pos_indices_list[:-1]  # do not care "O"
                token_label_precision_macro = tf_metrics.precision(token_label_ids, token_label_predictions,
                                                                   num_token_labels,
                                                                   pos_indices_list, average="macro")
                token_label_recall_macro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="macro")
                token_label_f_macro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels,
                                                    pos_indices_list,
                                                    average="macro")
                token_label_precision_micro = tf_metrics.precision(token_label_ids, token_label_predictions,
                                                                   num_token_labels,
                                                                   pos_indices_list, average="micro")
                token_label_recall_micro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="micro")
                token_label_f_micro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels,
                                                    pos_indices_list,
                                                    average="micro")
                token_label_loss = tf.metrics.mean(values=token_label_per_example_loss, weights=is_real_example)
                predicate_head_select_loss = tf.metrics.mean(values=predicate_head_select_loss)
                multi_head_selection_weight = is_real_example[:, tf.newaxis, tf.newaxis, tf.newaxis]

                precision_multi_head_selection = tf.metrics.precision_at_thresholds(true_predicate_matrix,
                                                                                    pred_predicate_matrix,
                                                                                    thresholds=[0.5],
                                                                                    weights=multi_head_selection_weight)
                recall_multi_head_selection = tf.metrics.recall_at_thresholds(true_predicate_matrix,
                                                                              pred_predicate_matrix,
                                                                              thresholds=[0.5],
                                                                              weights=multi_head_selection_weight)
                return {
                    "predicate_head_select_loss": predicate_head_select_loss,
                    "eval_token_label_precision(macro)": token_label_precision_macro,
                    "eval_token_label_recall(macro)": token_label_recall_macro,
                    "eval_token_label_f(macro)": token_label_f_macro,
                    "eval_token_label_precision(micro)": token_label_precision_micro,
                    "eval_token_label_recall(micro)": token_label_recall_micro,
                    "eval_token_label_f(micro)": token_label_f_micro,
                    "eval_token_label_loss": token_label_loss,
                    "eval_multi_head_selection_precision": precision_multi_head_selection,
                    "eval_multi_head_selection_recall": recall_multi_head_selection
                }

            eval_metrics = (metric_fn,
                            [predicate_matrix, predicate_head_probabilities, predicate_head_select_loss, token_label_per_example_loss,
                             token_label_ids, token_label_logits, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "predicate_head_probabilities": predicate_head_probabilities,
                    "predicate_head_predictions": predicate_head_predictions,
                    "token_label_predictions": token_label_predictions},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ske_2019": SKE_2019_Subject_Relation_Object_extraction_Processor,
        "ske_2019_v2": SKE_2019_Subject_Relation_Object_extraction_Processor_V2,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not any([FLAGS.do_train, FLAGS.do_eval, FLAGS.do_predict, FLAGS.do_interactive_predict, FLAGS.do_export]):
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_predict`, `do_interactive_predict' or `do_export` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    token_label_list = processor.get_token_labels()
    predicate_label_list = processor.get_predicate_labels()

    num_token_labels = len(token_label_list)
    num_predicate_labels = len(predicate_label_list)

    token_label_id2label = {}
    for (i, label) in enumerate(token_label_list):
        token_label_id2label[i] = label
    predicate_label_id2label = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_id2label[i] = label

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_token_labels=num_token_labels,
        num_predicate_labels=num_predicate_labels,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, token_label_list, predicate_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            num_predicate_label=len(predicate_label_list),
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, token_label_list, predicate_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            num_predicate_label=len(predicate_label_list),
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, token_label_list, predicate_label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            num_predicate_label=len(predicate_label_list),
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        token_label_output_predict_file = os.path.join(FLAGS.output_dir, "token_label_predictions.txt")
        predicate_output_predict_file = os.path.join(FLAGS.output_dir, "predicate_head_predictions.txt")
        predicate_output_predict_id_file = os.path.join(FLAGS.output_dir, "predicate_head_predictions_id.txt")
        predicate_head_probabilities_file = os.path.join(FLAGS.output_dir, "predicate_head_probabilities.txt")
        with open(token_label_output_predict_file, "w", encoding='utf-8') as token_label_writer, \
            open(predicate_output_predict_file, "w", encoding='utf-8') as predicate_head_predictions_writer, \
            open(predicate_output_predict_id_file, "w", encoding='utf-8') as predicate_head_predictions_id_writer, \
            open(predicate_head_probabilities_file, "w", encoding='utf-8') as predicate_head_probabilities_writer:
            num_written_lines = 0
            tf.logging.info("***** token_label predict and predicate labeling results *****")
            for (i, prediction) in enumerate(result):
                token_label_prediction = prediction["token_label_predictions"]
                predicate_head_predictions = prediction["predicate_head_predictions"]
                predicate_head_probabilities = prediction["predicate_head_probabilities"]
                if i >= num_actual_predict_examples:
                    break
                token_labels = [token_label_id2label[id_] for id_ in token_label_prediction]
                token_label_output_line = " ".join(token_labels) + "\n"
                token_label_writer.write(token_label_output_line)
                predicate_head_predictions_flatten = predicate_head_predictions.flatten()
                # predicate_head_predictions_line = " ".join(predicate_head_prediction)
                positive_triplets = np.argwhere(predicate_head_probabilities > 0.5)

                # Re-do input feature conversion to restore model input
                current_input_feature = convert_single_example(i, predict_examples[i], token_label_list, predicate_label_list,
                                                               FLAGS.max_seq_length, tokenizer, len(predicate_label_list))
                tokens = tokenizer.convert_ids_to_tokens(current_input_feature.input_ids)
                l, r = 1, len(tokens)
                for idx, token in enumerate(tokens):
                    if token == "[SEP]":
                        r = idx
                        break
                human_readable_triplets = []
                for loc_i, loc_j, id_ in positive_triplets:
                    subject = get_spans(loc_i, token_labels, tokens)
                    object_ = get_spans(loc_j, token_labels, tokens)
                    is_valid_triplet = subject is not None and \
                                       object_ is not None and \
                                       (l <= loc_i < r) and \
                                       (l <= loc_j < r)
                    if not is_valid_triplet:
                        continue
                    human_readable_triplets.append((subject, predicate_label_id2label[id_], object_))
                predicate_head_predictions_line = " ".join("{}-[{}]->{}".format(*spo_triplet) for spo_triplet in human_readable_triplets) + "\n"
                predicate_head_predictions_writer.write(predicate_head_predictions_line)

                predicate_head_predictions_id_line = " ".join(
                    str(id_) for id_ in predicate_head_predictions_flatten) + "\n"
                predicate_head_predictions_id_writer.write(predicate_head_predictions_id_line)

                # predicate_head_probabilities_flatten = predicate_head_probabilities.flatten()
                # predicate_head_probabilities_line = " ".join(str(prob) for prob in predicate_head_probabilities_flatten) + "\n"
                # predicate_head_probabilities_writer.write(predicate_head_probabilities_line)

                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    if FLAGS.do_interactive_predict:
        current_input_feature: InputFeatures = None

        def input_stream():
            nonlocal current_input_feature
            idx = 0
            while True:
                print("Input sentence:")
                line = sys.stdin.readline().rstrip()
                if not line:
                    print("Empty input")
                    continue
                tokens_with_oov = tokenizer.tokenize(line, convert_oov=True)
                example = InputExample("example-{}".format(idx), " ".join(tokens_with_oov))
                feature = convert_single_example(idx, example, token_label_list, predicate_label_list,
                                                 FLAGS.max_seq_length, tokenizer, len(predicate_label_list))
                current_input_feature = feature
                predicate_matrix_indices = np.array(feature.predicate_matrix_ids[0]).reshape([-1, 3])
                predicate_matrix_values = np.array(feature.predicate_matrix_ids[1]).reshape([-1])

                data = {
                    "input_ids": feature.input_ids,
                    "input_mask": feature.input_mask,
                    "segment_ids": feature.segment_ids,
                    "token_label_ids": feature.token_label_ids,
                    "sparse_predicate_matrix_indices": predicate_matrix_indices,
                    "sparse_predicate_matrix_values": predicate_matrix_values,
                    "is_real_example": 1.
                }
                yield data
                idx += 1

        result = estimator.predict(input_fn=
                                   interactive_input_fn_builder(input_stream,
                                                                seq_length=FLAGS.max_seq_length,
                                                                num_predicate_label=len(predicate_label_list),))

        tf.logging.info("***** token_label predict and predicate labeling results *****")
        for (i, prediction) in enumerate(result):
            tokens = tokenizer.convert_ids_to_tokens(current_input_feature.input_ids)
            l, r = 1, len(tokens)
            for idx, token in enumerate(tokens):
                if token == "[SEP]":
                    r = idx
                    break
            token_label_prediction = prediction["token_label_predictions"]
            predicate_head_predictions = prediction["predicate_head_predictions"]
            predicate_head_probabilities = prediction["predicate_head_probabilities"]
            token_labels = [token_label_id2label[id_] for id_ in token_label_prediction]
            assert len(tokens) == len(token_labels)
            token_label_output_line = " ".join(token_labels) + "\n"
            positive_triplets = np.argwhere(predicate_head_probabilities > 0.5)
            for loc_i, loc_j, id_ in positive_triplets:
                subject = get_spans(loc_i, token_labels, tokens)
                object_ = get_spans(loc_j, token_labels, tokens)
                is_valid_triplet = subject is not None and \
                                   object_ is not None and\
                                   (l <= loc_i < r) and \
                                   (l <= loc_j < r)
                if not is_valid_triplet:
                    print("Invalid triplet:")
                print("{}-[{}]->{}".format(subject, predicate_label_id2label[id_], object_))

    if FLAGS.do_export:
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(FLAGS.output_dir, "export"))
        with tf.Graph().as_default():
            serialized_tf_examples = tf.placeholder(tf.string, name="tf_example")
            seq_length = FLAGS.max_seq_length
            name_to_features = {
                "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
                "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
                "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
                "token_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
                "predicate_matrix": tf.io.FixedLenFeature([seq_length, seq_length, num_predicate_labels], tf.float32),
                "is_real_example": tf.FixedLenFeature([], tf.int64),
            }
            # deserialize
            tf_examples = tf.parse_example(serialized_tf_examples, name_to_features)
            # Rename and optionally cast `dtype` of tensors
            use_int32 = True
            predicate_matrix = tf.identity(tf_examples["predicate_matrix"], name="predicate_matrix")
            if use_int32:
                input_ids = tf.cast(tf_examples["input_ids"], tf.int32, name="input_ids")
                input_mask = tf.cast(tf_examples["input_mask"], tf.int32, name="input_mask")
                segment_ids = tf.cast(tf_examples["segment_ids"], tf.int32, name="segment_ids")
                token_label_ids = tf.cast(tf_examples["token_label_ids"], tf.int32, name="token_label_ids")
            else:
                input_ids = tf.identity(tf_examples["input_ids"], name="input_ids")
                input_mask = tf.identity(tf_examples["input_mask"], name="input_mask")
                segment_ids = tf.identity(tf_examples["segment_ids"], name="segment_ids")
                token_label_ids = tf.identity(tf_examples["token_label_ids"], name="token_label_ids")

            for name in sorted(tf_examples.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, tf_examples[name].shape))
            use_one_hot_embeddings = FLAGS.use_tpu
            is_training = False
            (total_loss,
             predicate_head_select_loss, predicate_head_probabilities, predicate_head_predictions,
             token_label_loss, token_label_per_example_loss, token_label_logits,
             token_label_predictions) = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids,
                token_label_ids, predicate_matrix, num_token_labels, num_predicate_labels,
                use_one_hot_embeddings)

            tvars = tf.trainable_variables()
            tf.logging.info("***** Loading Model to be Exported *****")
            if tf.train.latest_checkpoint(FLAGS.output_dir):
                init_checkpoint = FLAGS.output_dir
            else:
                tf.logging.info("Could not find checkpoint specified in `output_dir`, "
                                "loading model from `init_checkpoint`.")
                init_checkpoint = FLAGS.init_checkpoint
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if FLAGS.use_tpu:
                raise NotImplemented("Exporting model for TPU is not implemented/tested yet.")
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)
            default_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "input_ids": tf.saved_model.utils.build_tensor_info(input_ids),
                        "input_mask": tf.saved_model.utils.build_tensor_info(input_mask),
                        "segment_ids": tf.saved_model.utils.build_tensor_info(segment_ids)
                    },
                    outputs={
                        "token_label_logits": tf.saved_model.utils.build_tensor_info(token_label_logits),
                        "predicate_head_probabilities": tf.saved_model.utils.build_tensor_info(predicate_head_probabilities),
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "input_ids": tf.saved_model.utils.build_tensor_info(input_ids),
                        "input_mask": tf.saved_model.utils.build_tensor_info(input_mask),
                        "segment_ids": tf.saved_model.utils.build_tensor_info(segment_ids)
                    },
                    outputs={
                        "token_label_predictions": tf.saved_model.utils.build_tensor_info(token_label_predictions),
                        "predicate_head_probabilities": tf.saved_model.utils.build_tensor_info(predicate_head_probabilities),
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                     signature_def_map={
                                                         tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_signature,
                                                         "predict": prediction_signature
                                                     },
                                                     main_op=tf.tables_initializer(),
                                                     strip_default_attrs=True)
                builder.save()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
