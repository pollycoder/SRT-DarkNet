import os
import random
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.backend import set_session
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D
from keras import initializers


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = False  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

# This is tuned hyper-parameters
alpha = 0.1
batch_size = 128
# emb_size = 64
emb_size = 5000
number_epoch = 50
website_num = 1000

UNKNOWN = 1000
PADDING = 1001

# setting
multi_tab = 5
feature_dimension = 5000

os.environ["CUDA_VISIBLE_DEVICES"]="9"

base_dir = "/data/users/zhaoxiyuan/XMC-WFP-local/data/100*10"
train_file = os.path.join(base_dir, "train.pickle")

description = 'Multiple_Model'


alpha_value = float(alpha)


# 提取训练特征表示器的数据
def extract_train(tot_data):
    instances = np.zeros((len(tot_data), feature_dimension), dtype = float)
    # 保存每组多tab标签的样例
    label_instance = {}
    # 保存每个instance的样例
    index_label = []
    # 保存单个标签下的样例
    single_label_instance = {}
    tot_num = 0
    for i in range(len(tot_data)):
        data = tot_data[i]
        instances[i] = data[:feature_dimension]
        # 此处把标签按照从小到大的顺序排序
        temp_label = []
        for j in range(multi_tab):
            single_label = int(data[20000+j])
            if single_label == UNKNOWN:
                continue
            temp_label.append(single_label)
            if single_label not in single_label_instance:
                single_label_instance[single_label] = []
            single_label_instance[single_label].append(i)

        if len(temp_label) == 0:
            index_label.append(())
            continue

        temp_label = tuple(sorted(temp_label))
        # 统计每个标签组合下的样本
        if temp_label not in label_instance:
            label_instance[temp_label] = []
        label_instance[temp_label].append(i)
        index_label.append(temp_label)
    return instances, label_instance, single_label_instance, index_label


#  提取用于训练树上分类器样例
def extract_others(tot_data):
    instances = np.zeros((len(tot_data), feature_dimension), dtype = float)
    labels1 = np.zeros((len(tot_data), multi_tab), dtype=int)
    for i in range(len(tot_data)):
        data = tot_data[i]
        instances[i] = data[:feature_dimension]
        for j in range(multi_tab):
            labels1[i][j] = int(data[feature_dimension+j])
    # with open(file, "wb") as file:
    #     pickle.dump(labels1, file)
    return instances




# ================================================================================
# This part is to prepare the files' index for geenrating triplet examples
# and formulating each epoch inputs
# # Extract all folders' names

# # Each given folder name (URL of each class), we assign class id
# # e.g. {'adp.com' : 23, ...}
# name_to_classid = {d:i for i,d in enumerate(dirs)}

# # Just reverse from previous step
# # Each given class id, show the folder name (URL of each class)
# # e.g. {23 : 'adp.com', ...}
# classid_to_name = {v:k for k,v in name_to_classid.items()}

train_data = pd.read_pickle(train_file)
train_data = np.array(pd.DataFrame(train_data).astype(float))
print(f">>>>>>>>>>>>{train_data.shape}")

train_instances, label_instance, single_label_instance, index_label = extract_train(train_data)

train_instances = train_instances[:, :, np.newaxis]

# left_train1, left_train2 = train_test_split(left_train, test_size=0.5, random_state=42)
# label_file = "/data/users/zhaoxiyuan/XMC-WFP-local/result/webpage/1tab/train_classifier_label.pkl"
# classifier_file = os.path.join(base_dir, "classifier.pickle")
# classifier_data = pd.read_pickle(classifier_file)
# classifier_data = np.array(pd.DataFrame(classifier_data).astype(float))
# print(f">>>>>>>>>>>>{classifier_data.shape}")
# # classifier_instance = extract_others(classifier_data)
# classifier_instance = classifier_data[:, :feature_dimension]
# classifier_instance = classifier_instance[:, :, np.newaxis]

# train_label_instance = extract_train_label(left_train2)
# train_label_instance = train_label_instance[:, :, np.newaxis]

# label_file = "/data/users/zhaoxiyuan/XMC-WFP-local/result/webpage/1tab/test_label.pkl"
# test_file = os.path.join(base_dir, "test.pickle")
# test_data = pd.read_pickle(test_file)
# test_data = np.array(pd.DataFrame(test_data).astype(float))
# print(f">>>>>>>>>>>>{test_data.shape}")
# # test_instances = extract_others(test_data)
# test_instances = test_data[:, :feature_dimension]
# test_instances = test_instances[:, :, np.newaxis]


def build_pos_pairs_for_id(instances): # 给出一个标签组合下的所有Instance
    # pos_pairs = []
    # for i in range(len(instances)-1):
    #     j = random.randint(i+1,len(instances)-1)
    #     pos_pairs.append((instances[i], instances[j]))
    pos_pairs = [(instances[i], instances[j]) for i in range(len(instances)) for j in range(i+1, len(instances))]
    random.shuffle(pos_pairs)
    return pos_pairs
def build_positive_pairs():
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for _class in label_instance.keys():
        pos = build_pos_pairs_for_id(label_instance[_class])
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]

Xa_train, Xp_train = build_positive_pairs()

# Gather the ids of all network traces that are used for training
# This just union of two sets set(A) | set(B)
all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
print ("X_train Anchor: ", Xa_train.shape)
print ("X_train Positive: ", Xp_train.shape)

# Build a loss which doesn't take into account the y_true, as# Build
# we'll be passing only 0
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


@tf.function
def is_equal(a, b):
    aeqb = tf.equal(a,b)
    aeqb_int = tf.cast(aeqb, tf.int32)
    result = tf.equal(tf.reduce_sum(aeqb_int),tf.reduce_sum(tf.ones_like(aeqb_int)))
    return result


# The real loss is here，改进了triplet loss
@tf.function
def cosine_triplet_loss(X):
    _alpha = alpha_value
    positive_sim, negative_all_sim, negative_1_sim, negative_2_sim, negative_3_sim, negative_4_sim = X
    # 1个标签
    if is_equal(negative_all_sim, negative_1_sim):
        losses = K.maximum(0.0, negative_all_sim - positive_sim + _alpha)
    # 2个标签
    elif is_equal(negative_1_sim, negative_2_sim):
        losses = K.maximum(0.0, 1/2 * negative_1_sim + negative_all_sim - 1.5 * positive_sim + 1.5 * _alpha) / 1.5
 # 3个标签
    elif is_equal(negative_1_sim, negative_3_sim):
        losses = K.maximum(0.0, 2/3 * negative_1_sim + 1/3 * negative_2_sim + negative_all_sim - 2 * positive_sim + 2 * _alpha) / 2.0

    # 4个标签
    elif is_equal(negative_1_sim, negative_4_sim):
        losses = K.maximum(0.0, 3/4 * negative_1_sim + 2/4 * negative_2_sim + 1/4 * negative_3_sim + negative_all_sim - 2.5 * positive_sim + 2.5 * _alpha) / 2.5

    # 5个标签
    else:
        losses = K.maximum(0.0, 4/5 * negative_1_sim + 3/5 * negative_2_sim + 2/5 * negative_3_sim + 1/5 * negative_4_sim + negative_all_sim - 3 * positive_sim + 3 * _alpha) / 3.0
    return K.mean(losses)

# ------------------- Hard Triplet Mining -----------
# Naive way to compute all similarities between all network traces.

def build_similarities(conv, all_imgs):
    embs = conv.predict(all_imgs)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims

def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs)), random.sample(neg_imgs_idx,len(anc_idxs)), random.sample(neg_imgs_idx,len(anc_idxs)), random.sample(neg_imgs_idx,len(anc_idxs)), random.sample(neg_imgs_idx)
    final_1_neg = []  # 有一个相同标签
    final_2_neg = []  # 有两个相同标签
    final_3_neg = []  # 有三个相同标签
    final_4_neg = []  # 有四个相同标签
    final_all_neg = []  # 全部相同标签

    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):

        anchor_class = index_label[anc_idx]
        #positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_all_ids = intersect(neg_imgs_idx, possible_ids)
        neg_imgs_half_idx = []
        # 负样例取有相似标签但不完全相同的样例
        for elem in anchor_class:
            neg_imgs_half_idx.extend(single_label_instance[elem])
        possible_half_ids = intersect(neg_imgs_half_idx, possible_ids)
        appended_1 = False
        appended_2 = False
        appended_3 = False 
        appended_4 = False
        appended_all = False
        # 2tab时，将负样例2，负样例3，负样例4都选为1个标签相同的负样例，这样也可以用于后续判断是否为2tab，以下同理
        if len(anchor_class) == 2:
            appended_2 = True
            appended_3 = True
            appended_4 = True
        if len(anchor_class) == 3:
            appended_3 = True
            appended_4 = True
        if len(anchor_class) == 4:
            appended_4 = True
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_all_neg = random.choice(possible_all_ids)
            if not appended_all and not intersect(index_label[idx_all_neg], anchor_class):
                # 单标签时单独处理，2，3，4，all都选相同的负样例
                if len(anchor_class) == 1:
                    final_all_neg.append(idx_all_neg)
                    appended_all = True
                    final_1_neg.append(idx_all_neg)
                    appended_1 = True
                    final_2_neg.append(idx_all_neg)
                    appended_2 = True
                    final_3_neg.append(idx_all_neg)
                    appended_3 = True
                    final_4_neg.append(idx_all_neg)
                    appended_4 = True
                    break
                final_all_neg.append(idx_all_neg)
                appended_all = True
            if len(anchor_class) == 1:
                continue
            idx_half_neg = random.choice(possible_half_ids)
            # append1个标签相同的负样例
            if not appended_1 and len(intersect(anchor_class, index_label[idx_half_neg])) == 1:
                final_1_neg.append(idx_half_neg)
                appended_1 = True
            # append2个标签相同的负样例
            if not appended_2 and len(intersect(anchor_class, index_label[idx_half_neg])) == 2:
                final_2_neg.append(idx_half_neg)
                appended_2 = True
            # append3个标签相同的负样例
            if not appended_3 and len(intersect(anchor_class, index_label[idx_half_neg])) == 3:
                final_3_neg.append(idx_half_neg)
                appended_3 = True
            # append4个标签相同的负样例
            if not appended_4 and len(intersect(anchor_class, index_label[idx_half_neg])) == 4:
                final_4_neg.append(idx_half_neg)
                appended_4 = True

            if appended_all and appended_1 and appended_2 and appended_3 and appended_4:
                break
        if not appended_all:
            final_all_neg.append(random.choice(neg_imgs_idx))
        if not appended_1:
            final_1_neg.append(random.choice(neg_imgs_half_idx))
        if not appended_2:
            final_2_neg.append(random.choice(neg_imgs_half_idx))
        if not appended_3:
            final_3_neg.append(random.choice(neg_imgs_half_idx))
        if not appended_4:
            final_4_neg.append(random.choice(neg_imgs_half_idx))
        if len(anchor_class) == 2:
            final_2_neg.append(final_1_neg[-1])
            final_3_neg.append(final_1_neg[-1])
            final_4_neg.append(final_1_neg[-1])
        if len(anchor_class) == 3:
            final_3_neg.append(final_1_neg[-1])
            final_4_neg.append(final_1_neg[-1])
        if len(anchor_class) == 4:
            final_4_neg.append(final_1_neg[-1])


    return final_all_neg, final_1_neg, final_2_neg, final_3_neg, final_4_neg


class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.neg_traces_idx = neg_traces_idx
        self.num_samples = Xa_train.shape[0]
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n_a, traces_n_1, traces_n_2, traces_n_3, traces_n_4 = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)
            # neg_ins = np.stack((np.array(traces_a, dtype=float), np.array(traces_n_1, dtype=float),
            #                     np.array(traces_n_2, dtype=float), np.array(traces_n_3, dtype=float),
            #                     np.array(traces_n_4, dtype=float)), axis=1)
            yield ([self.traces[traces_a],
                    self.traces[traces_p],
                    self.traces[traces_n_a],
                    self.traces[traces_n_1],
                    self.traces[traces_n_2],
                    self.traces[traces_n_3],
                    self.traces[traces_n_4],
                    # neg_ins
                    ],
                   np.zeros(shape=(traces_a.shape[0]))
                   )

# Training the Triplet Model
from DF_model import DF
shared_conv2 = DF(input_shape=(feature_dimension,1), emb_size=emb_size)

anchor = Input((feature_dimension, 1), name='anchor')
positive = Input((feature_dimension, 1), name='positive')
all_negative = Input((feature_dimension, 1), name='all negative')
negative_one = Input((feature_dimension, 1), name='1 label same')
negative_two = Input((feature_dimension, 1), name='2 label same')
negative_three = Input((feature_dimension, 1), name='3 label same')
negative_four = Input((feature_dimension, 1), name='4 label same')





a = shared_conv2(anchor)
p = shared_conv2(positive)
n_a = shared_conv2(all_negative)
n_1 = shared_conv2(negative_one)
n_2 = shared_conv2(negative_two)
n_3 = shared_conv2(negative_three)
n_4 = shared_conv2(negative_four)

# 将选取的负样例直接输出
# neg_ins = Conv1D(filters=1, kernel_size=1, kernel_initializer="ones")

# The Dot layer in Keras now supports built-in Cosine similarity using the normalize = True parameter.
# From the Keras Docs:
# keras.layers.Dot(axes, normalize=True)
# normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product.
#  If set to True, then the output of the dot product is the cosine proximity between the two samples.
pos_sim = Dot(axes=-1, normalize=True)([a,p])
neg_all_sim = Dot(axes=-1, normalize=True)([a,n_a])
neg_1_sim = Dot(axes=-1, normalize=True)([a,n_1])
neg_2_sim = Dot(axes=-1, normalize=True)([a,n_2])
neg_3_sim = Dot(axes=-1, normalize=True)([a,n_3])
neg_4_sim = Dot(axes=-1, normalize=True)([a,n_4])

# customized loss
loss = Lambda(cosine_triplet_loss,
              output_shape=(1,))(
             [pos_sim,neg_all_sim,neg_1_sim,neg_2_sim,neg_3_sim,neg_4_sim])

model_triplet = Model(
    inputs=[anchor, positive, all_negative, negative_one, negative_two, negative_three, negative_four],
    outputs=loss)

opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model_triplet.compile(loss=identity_loss, optimizer=opt)

# At first epoch we don't generate hard triplets
gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, train_instances, all_traces_train_idx, None)
nb_epochs = number_epoch
csv_logger = CSVLogger('log/Training_Log_%s.csv'%description, append=True, separator=';')
for epoch in range(nb_epochs):
    print("built new hard generator for epoch "+str(epoch))

    model_triplet.fit_generator(generator=gen_hard.next_train(),
                    steps_per_epoch=Xa_train.shape[0] // batch_size,
                    epochs=1, verbose=1, callbacks=[csv_logger])

    # 保存每轮训练好的模型
    network_file = f"/data/users/zhaoxiyuan/XMC-WFP-local/result/100*10/ML/networks/df_{epoch}.h5"
    shared_conv2.save(network_file)
    gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, train_instances, all_traces_train_idx, shared_conv2)


    #For no semi-hard_triplet
    #gen_hard = HardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
# shared_conv2.save('trained_model/%s.h5'%description)
# res_file = "/data/users/zhaoxiyuan/XMC-WFP-local/result/webpage/1tab_/representation_5000d_train.pkl"
# with open(res_file, "wb") as file:
#     pickle.dump(shared_conv2.predict(train_instances), file)

# res_file = "/data/users/zhaoxiyuan/XMC-WFP-local/result/webpage/1tab_/representation_5000d_train_tree.pkl"
# with open(res_file, "wb") as file:
#     pickle.dump(shared_conv2.predict(classifier_instance), file)


# # res_file = "/data/users/zhaoxiyuan/XMC-WFP-local/result/7networks/2tabs/representation_5000d_train_label.pkl"
# # with open(res_file, "wb") as file:
# #     pickle.dump(shared_conv2.predict(train_label_instance), file)


# res_file = "/data/users/zhaoxiyuan/XMC-WFP-local/result/webpage/1tab_/representation_5000d_test.pkl"
# with open(res_file, "wb") as file:
#     pickle.dump(shared_conv2.predict(test_instances), file)

