#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@desc:   简单的skip-gram模型实现的word2vec
@time:   2017/06/19 20：48
@author: liuluxin(0_0mirror@sina.com)
@param:
@param:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import Counter
from tensorflow.contrib.tensorboard.plugins import projector

import math
import sys, os
import random
import zipfile
import sys
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import ConfigParser

line_split = ""
qa_split = ""
word_split = ""
config_path = sys.argv[1]

print("配置路径为:{}".format(config_path))
'''
@desc: 得到路径配置
@format: [word2vec] train_data=xxxx
'''
def get_config(section, key):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    return config.get(section, key)

'''
@desc: 得到路径配置
@format: [word2vec] train_data=xxxx
'''
def get_config_int(section, key):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    return config.getint(section, key)

IS_DEBUG=get_config_int("word2vec","debug")

'''
@desc: 从配置中读取，并构建图所需的元素
'''
batch_size = get_config_int("word2vec", "batch_size")
embedding_size = get_config_int("word2vec", "embedding_size")  
skip_window = get_config_int("word2vec", "skip_window")  
num_skips = get_config_int("word2vec", "num_skips")  

valid_size = get_config_int("word2vec", "valid_size")  
valid_window = get_config_int("word2vec", "valid_window")
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = get_config_int("word2vec", "num_sampled")  
save_step_num = get_config_int("word2vec", "save_step_num")

'''
@desc: 从行从split出word并将低频词和停用词(删除概率P(Wi)=1-sqrt(t/frequent(Wi))都平滑掉
@param: freq:小于freq次数的词都会被删除
        del_threshold: 大于这个阈值的词会被作为停用词被删除
'''
def get_pickle_file():
    #read_raw_words()
    pickle_file = open(get_config("word2vec", "pickle_path"),"rb")
    vocab_2_idx = pickle.load(pickle_file)
    idx_2_vocab = {i:w for w,i in vocab_2_idx.items()}
    pickle_file.close()
    qa_sents = open(get_config("word2vec","qa_list_file")).readlines()
    print("读取文件成功，总词表长度为{0}, 句子个数为{1}".format(len(idx_2_vocab),len(qa_sents)))
    return len(vocab_2_idx), vocab_2_idx, idx_2_vocab,qa_sents, len(qa_sents)

vocab_size,vocab_2_idx, idx_2_vocab, qa_sents, all_line_num = get_pickle_file()

'''
@desc: 从qa文件的每行中，在windowsize的窗口内随机产生batch数据
@param: line_begin: 开始采样的句子位置
        line_end: 结束采样的句子位置
        num_skips:每个词的重用次数，取决于window的大小
        skip_window: 采样词的左右窗口大小（即决定了进行几gram的采样)skip_windows*2=num_skips
'''
line_idx=0
word_idx=0
def generate_batch(batch_size, num_skips, skip_window):
    assert num_skips <= 2 * skip_window
    UNK_idx = vocab_2_idx["UNK"]
    batch_list = []
    label_list = []
    #根据指定的行号从q和a的sentence中取出需要的batch
    while len(batch_list) < batch_size:
        global line_idx,word_idx
        line_idx +=1 
        if line_idx >= all_line_num :
            line_idx = 0
        query_list = []
        query_list.extend(qa_sents[line_idx].split(word_split))
        for idx in range(word_idx,len(query_list)):
            if query_list[idx] != UNK_idx:
                    input_id = query_list[idx]
                    target_window = np.random.randint(1, skip_window + 1)
                    start = max(0, idx - target_window)
                    end = min(len(query_list) - 1, idx + target_window)
                    for i in range(start, end):
                        if idx != i:
                            output_id = query_list[i]
                            batch_list.append(input_id)
                            label_list.append(output_id)
                            if len(batch_list)== batch_size:
				if IS_DEBUG==True:
				    print("Generate batch size is {}".format(len(batch_list)))
                                batchs = np.array(batch_list, dtype=np.int32)
                                batchs = batchs.reshape([batch_size])
                                labels = np.array(label_list, dtype=np.int32)
                                labels = labels.reshape([batch_size,1])
                                word_idx = idx + 1
				if IS_DEBUG==True:
				    print("Generate batch size is {}".format(len(batch_list)))
                                    print(batch_list)
                                return  batchs,labels
        if word_idx >= len(query_list):
            word_idx = 0

test_batch, test_label= generate_batch(batch_size, num_skips=2, skip_window=1)
for i in range(batch_size):
    print(test_batch[i], idx_2_vocab[test_batch[i]],
          '->', test_label[i, 0], idx_2_vocab[test_label[i, 0]])

graph = tf.Graph()
with graph.as_default():
    for d in ['/cpu:0']:
        with tf.device(d):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="embeddings")
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)), name="nec_weight")
            nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")

        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocab_size))
        loss_log = tf.summary.scalar('loss', loss)
        biases_log = tf.summary.histogram("basis",nce_biases)
        weight_log = tf.summary.histogram("weight",nce_weights)
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # 计算候选embedding的cosine相似度
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        norm_log = tf.summary.histogram("norm",norm)
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
        merged_summary_op = tf.summary.merge([loss_log,weight_log,biases_log,norm_log])

'''
@desc.绘制图像存储到指定png文件
'''
def plot_with_labels(low_dim_embs, labels, filename,log_writer):
    assert low_dim_embs.shape[0] >= len(labels), 'more labels than embeddings'
    plt.figure()  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label.encode("utf-8"),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom', fontproperties=myfont)

    plt.savefig(filename)
    file = open(filename, 'rb')
    data = file.read()
    file.close()


'''
@desc: 正式训练流程
'''
num_steps = get_config_int("word2vec", "num_steps")
#初始化所有数据和存储
log_dir =  get_config("word2vec","log_dir")
print('Begin Training')
with tf.Session(graph=graph) as session:
    model_path = get_config("word2vec","model_path")
    saver = tf.train.Saver()
    # 存在就从模型中恢复变量
    if os.path.exists(model_path):
        saver.restore(session, model_path)
    # 不存在就初始化变量
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, session.graph)

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs = []
        batch_labels = []
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val


        # 每200次迭代输出一次损失 保存一次模型
        if step % save_step_num == 0:
            if step > 0:
                average_loss /= 2000
            average_loss = 0
            save_path = saver.save(session, model_path, global_step=step)
            print("模型保存:{0}\t当前损失:{1}".format(model_path,  loss_val))

        # 每隔100次迭代，保存一次日志
        if step % 100 == 0:
            summary_str = session.run(merged_summary_op,feed_dict=feed_dict )
            summary_writer.add_summary(summary_str, step)

        # 每step_num词隔迭代输出一次指定词语的最近邻居
        if step % 100 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = idx_2_vocab[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = idx_2_vocab[nearest[k]]
                    log_str = '%s %s(%d):%f' % (log_str, close_word, nearest[k],-sim[i,k])
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib as mpl

        #mpl.use('Agg')
        from matplotlib.font_manager import *
        import matplotlib.pyplot as plt

        myfont = FontProperties(
            fname="//data01/ai_rd/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf")
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 100
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [idx_2_vocab[i] for i in xrange(plot_only)]
        embs_pic_path = get_config("word2vec", "embs_pic_path")
        plot_with_labels(low_dim_embs, labels, embs_pic_path, summary_writer)

        file = open(embs_pic_path, 'rb')
        data = file.read()
        file.close()

        image = tf.image.decode_png(data, channels=4)
        image = tf.expand_dims(image, 0)

        # 添加到日志中
        summary_op = tf.summary.image("image1", image)

        # 运行并写入日志
        summary = session.run(summary_op)
        summary_writer.add_summary(summary)

    except ImportError:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')



