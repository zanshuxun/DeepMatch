

import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

data_path = "./"

unames = ['user_id','gender','age','occupation','zip']
user = pd.read_csv(data_path+'ml-1m/users.dat',sep='::',header=None,names=unames)
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(data_path+'ml-1m/ratings.dat',sep='::',header=None,names=rnames)
mnames = ['movie_id','title','genres']
movies = pd.read_csv(data_path+'ml-1m/movies.dat',sep='::',header=None,names=mnames)

data = pd.merge(pd.merge(ratings,movies),user)#.iloc[:10000]


sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
SEQ_LEN = 50

# 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
feature_max_idx = {}
for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1

    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

item_profile = data[["movie_id"]].drop_duplicates('movie_id')

user_profile.set_index("user_id", inplace=True)

user_item_list = data.groupby("user_id")['movie_id'].apply(list)
print(user_item_list)

train_set, test_set = gen_data_set(data, 0)

train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

# 2.count #unique features for each sparse field and generate feature config for sequence feature

embedding_dim = 32

user_feature_columns = [VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len')]

item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

# 3.Define Model and train

K.set_learning_phase(True)
import tensorflow as tf

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()

model = NARM(user_feature_columns, item_feature_columns, num_sampled=100, gru_hidden_units=(64,32))

model.compile(optimizer="adam", loss=sampledsoftmaxloss)  # "binary_crossentropy")

history = model.fit(train_model_input, train_label,  # train_label,
                    batch_size=512, epochs=20, verbose=1, validation_split=0.0, )

# 4. Generate user features for testing and full item features for retrieval
test_user_model_input = test_model_input
all_item_model_input = {"movie_id": item_profile['movie_id'].values}

user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

print(user_embs.shape)
print(item_embs.shape)

# 5. [Optional] ANN search by faiss  and evaluate the result

test_true_label = {line[0]:[line[2]] for line in test_set}

import numpy as np
import faiss
from tqdm import tqdm
from deepmatch.utils import recall_N

index = faiss.IndexFlatIP(embedding_dim)
# faiss.normalize_L2(item_embs)
index.add(item_embs)
# faiss.normalize_L2(user_embs)
D, I = index.search(np.ascontiguousarray(user_embs), 50)
s = []
hit = 0
for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    try:
        pred = [item_profile['movie_id'].values[x] for x in I[i]]
        filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=50)
        s.append(recall_score)
        if test_true_label[uid] in pred:
            hit += 1
    except:
        print(i)
print("")
print("recall", np.mean(s))
print("hit rate", hit / len(test_user_model_input['user_id']))


# In[ ]:




