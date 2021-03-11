#!/usr/bin/env python
# coding: utf-8

# # CoLab 실행 코드

# ## 1.필요 패키지

# In[1]:


import time
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from rdkit import Chem
import matplotlib.pyplot as plt

print(tf.__version__)
get_ipython().system('python --version')


# ---

# ## 2. 데이터 전처리

# ### 파일 위치
# smiles conversion
#         ㅣ
#           --jupyter_submit_code.ipynb
#         ㅣ
#           --checkpoint
#         ㅣ
#           --train.csv
#         ㅣ
#           --sample_submission.csv
#         ㅣ
#           --train
#                 l
#                   --train_0.png
#                 l
#                   --train_1.png
#                 l
#                   -- ...
#         ㅣ
#           --test
#                 l
#                   --test_0.png
#                 l
#                   --test_1.png
#                 l
#                   -- ...
# ### 제공된 데이터를 사용하여 학습

# In[2]:


img_file_path = []  # 훈련에 사용할 이미지의 경로가 들어갈 리스트
train_y = []  # 훈련에 사용할 SMILES 식이 들어갈 리스트

PATH = '.\\train\\'  # 훈련에 사용할 이미지가 있는 폴더 경로
train_csv = pd.read_csv('.\\train.csv')   # train에 필요한 csv파일 경로
print(train_csv)

# SMILES식의 앞 뒤로 문장의 시작과 끝을 알리는 기호를 더합니다.
train_csv.SMILES = train_csv.SMILES.apply(lambda x: '<' + x + '>')

for file_name in train_csv.file_name:
    img_file_path.append(PATH + file_name)

for smiles in train_csv.SMILES:
    train_y.append(smiles)

number_of_data_used = 908765  # 학습에 사용할 데이터 수, 제공된 데이터 모두 사용하였습니다.
# slice
img_file_path = img_file_path[:number_of_data_used]
train_y = train_y[:number_of_data_used]


# ### 입력에 맞게 이미지를 로드하는 함수 정의

# In[3]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img, image_path


# ### 빠른 train, test를 위해 Image의 feature를 추출하는 Model 정의
# imagenet으로 pretrain된 InceptionResNetV2를 이용하여 이미지의 feature 추출

# In[4]:


#imagenet으로 pretrain된 InceptionResNetV2 모델 사용
image_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

# image feature를 추출하는 model을 정의합니다.
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# ### 빠른 train을 위해 추출된 feature를 로컬디스크에 저장
# 추출된 (8x8x1536)의 feature를 (64x1536)로 reshape하여 저장합니다

# In[5]:


encode_train = sorted(set(img_file_path))

# 데이터를 불러옵니다.
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# dataset의 각 멤버에 사용자 지정 함수(load_image)를 적용합니다. 
image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

num = 0
for img, path in image_dataset:
    # image_features를 추출합니다.
    batch_features = image_features_extract_model(img)
    # (8, 8, 1536)의 feature를 (64, 1536)으로 reshape합니다.
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

    # train_1.png 그림을 train_1.png.npy 로 저장합니다.
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())
        num += 1

        if num % 10000 == 0:
            print(num)
print("feature save")


# ### SMILES식을 토큰화

# In[6]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, char_level=True) # 한 글자 단위로 tokenizer
tokenizer.fit_on_texts(train_y) # 단어 집합 생성
train_y = tokenizer.texts_to_sequences(train_y) # 단어를 숫자로 변환
# 가장 긴 문장을 기준으로 짧은 문장의 뒤에 0을 채워넣음
train_y = tf.keras.preprocessing.sequence.pad_sequences(train_y, padding='post')
print(train_y.shape)


# ### 데이터셋 전체를 train에 사용함으로 학습, 검증 데이터셋으로 분리하지 않고 섞어줍니다.

# In[7]:


img_name_train, cap_train = shuffle(img_file_path, train_y, random_state=42)


# ### 하이퍼 파라미터 및 학습에 필요한 변수 지정

# In[8]:


BATCH_SIZE = 128
BUFFER_SIZE = 1000
seq_max_num = train_y.shape[1]  # SMILES식 중 가장 긴 문장의 길이
dropout_rate = 0.1  # dropout 계수
d_model = 512  # 현 모델의 인코더와 디코더에서의 정해진 입력과 출력의 크기
dff = 2048  # feed forward network의 hidden layer의 크기
num_heads = 8  # 병렬 Attention의 개수
num_layers = 8  # 디코더를 하나의 Layer로 가정하였을 때, 디코더를 8개의 층을 쌓아 설계
vocab_size = len(tokenizer.word_index) + 1  # dic의 단어 개수


# ### 데이터셋 정의 함수

# In[9]:


def map_func(img_name, cap):
    """npy 데이터를 읽어옵니다."""
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


# ### 학습 데이터셋 준비
# BATCH_SIZE만큼 로컬에서 불러와 학습

# In[10]:


# (list, numpy.array)에서 데이터를 불러옵니다.
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# dataset의 각 멤버에 사용자 지정 함수(map_func)를 적용합니다. 
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
# shuffle()을 사용하여 설정된 epoch 마다 dataset 을 섞은 후, batch size 지정
dataset = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)
# 모델이 s스텝 훈련을 실행하는 동안 입력 파이프라인은 s+1스텝의 데이터를 읽어옵니다.
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ---

# ## 3. 모델 구축 및 훈련
# 제안하는 모델은 Transformer Model을 응용한 모델입니다.
# 
# Transformer Model의 Encoder를 Inception resnet v2 모델을 통과한 feature를 사용하도록 변형하였으며,
# 
# Decoder는 Transformer Model의 구조를 사용하였습니다.

# ### scaled_dot_product_attention 정의
# 여러 종류의 Attention 중 Scaled_dot_product_attention을 사용합니다.

# In[11]:


def scaled_dot_product_attention(q, k, v, mask):
    """
       scaled_dot_product_attention의 구현
    1. Attention Score를 구한다.
    2. 스케일링
    3. Attention Distribution를 구한다.
    4. Attention Distribution 행렬과 V 행렬을 곱한다.
    """
    # query 크기 : (batch_size, num_heads, q의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, k의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, v의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, k의 문장 길이)
    
    # 1. Q와 K의 곱. 즉 Attention Score 행렬.
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., q의 문장 길이, k의 문장 길이)
    
    # 2. 스케일링
    # dk의 루트값으로 나눠준다.
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    # 매우 작은 값을 곱하므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        # mask값에 -1e9라는 아주 작은 음수값을 곱한 후 어텐션 스코어 행렬에 더해준다
        scaled_attention_logits += (mask * -1e9) 

    # 3. Attention Distribution를 구한다.
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # attention weight : (batch_size, num_heads, q의 문장 길이, k의 문장 길이)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # 4. Attention 분포 행렬과 V 행렬을 곱한다.
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# ### Positional Encoding을 위한 함수 정의
# Transformer Model은 RNN을 사용하지 않아 각 단어들이 위치 정보를 가질 수 없습니다.
# 
# 그러므로 위치 정보를 부여하기 위해 Positional Encoding을 사용합니다.

# In[12]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

  
def positional_encoding(position, d_model):
    """
    Transformer는 사인 함수와 코사인 함수의 값을 임베딩 벡터에 더해줌으로써 단어의 위치 정보를 부여합니다.
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# ### Masking을 위한 함수 정의

# In[13]:


def create_padding_mask(seq):
    """
    정수 시퀀스에서 0인 경우에는 1로 변환하고, 그렇지 않은 경우에는 0으로 변환하는 함수
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, key의 문장 길이)


def create_look_ahead_mask(size):
    """
    디코더의 첫번째 sub layer에서 미래 토큰을 Mask하는 함수입니다.
    
    문장 행렬로 입력을 한번에 받으므로, 현재 시점의 단어를 예측하고자 할 때,
    미래 시점의 단어까지 참고하여 예측하는 일이 발생합니다.
    이를 방지하기 위해, 현재 시점보다 미래에 있는 단어는 look_ahead_mask를 적용하여 가립니다.
    """
    
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    """
    Decoder layer의 sub_layer에서 사용하는 mask를 생성합니다.
    첫 번째 sub_layer에서는 look_ahead_mask와 padding_mask의 기능을 함께 사용합니다.
    
    아래 논문의 Transformer Model의 두 번째 sub_layer에서는 padding_mask를 사용하지만, 
    제안하는 모델은 두 번째 sub_layer의 입력으로 image_feature가 사용되므로 padding_mask를 사용하지 않습니다.
    
    Transformer 논문 출처 : http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    """
    # padding_mask 생성
    dec_padding_mask = create_padding_mask(inp)
    train_y_padding_mask = create_padding_mask(tar)
    
    # look_ahead_mask 생성
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    # look_ahead_mask를 하더라도 padding_mask가 필요하므로 2개의 마스크를 결합시킵니다.
    combined_mask = tf.maximum(train_y_padding_mask, look_ahead_mask)

    return combined_mask, dec_padding_mask


# ### Position_Wise_Feed_Forward_Network 함수 정의
# Layer의 입력과 출력의 크기를 같게 하기위한 sub_layer입니다.

# In[14]:


def position_wise_feed_forward_network(d_model, dff):
    """
    다음 디코더의 입력으로 사용이 가능하게끔, 
    이 함수의 출력의 크기와 다음 레이어의 입력의 크기를 같게 해주는 함수입니다.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# ### MultiHeadAttention 정의
# Attention을 병렬로 수행하기 위해 Multi Head Attention을 수행합니다.

# In[15]:


class MultiHeadAttention(tf.keras.layers.Layer):
    """
       Multi Head Attention의 구현
    1. WQ, WK, WV에 해당하는 d_model 크기의 Dense layer를 통과시킨다.
    2. 지정된 num_heads 만큼 나눈다.
    3. scaled_dot_product_attention
    4. 나눈 헤드들을 concat
    5. WO에 해당하는 Dense layer 통과
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 병렬 수행할 개수
        self.d_model = d_model  # 현 모델의 인코더와 디코더에서의 정해진 입력과 출력의 크기

        # d_model과 num_heads를 나눈 값이 0이 아니면 AssertionError가 발생합니다.
        # 나눈 값이 0이 되어야 하므로 assert를 사용해 값을 보증합니다.
        assert d_model % self.num_heads == 0  
        
        # d_model을 num_heads로 나눈 값.
        self.depth = d_model // self.num_heads
        
        # WQ, WK, WV : Q, K, V 행렬을 만들기 위한 가중치 행렬
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ num_heads 개수만큼 q, k, v를 split하는 함수 """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        q = self.wq(q)  # (batch_size, q의 문장 길이, d_model)
        k = self.wk(k)  # (batch_size, k의 문장 길이, d_model)
        v = self.wv(v)  # (batch_size, v의 문장 길이, d_model)

        # 2. 헤드 나누기
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, q의 문장 길이, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, k의 문장 길이, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, v의 문장 길이, depth)

        # 3. 스케일드 닷 프로덕트 어텐션
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # 4. 헤드 연결(concatenate)하기
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # 5. WO에 해당하는 밀집층 지나기
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# ### Model의 Encoder 정의
# Inception-resnet-v2에서 추출된 feature가 입력으로 들어갑니다.

# In[16]:


class Encoder(tf.keras.Model):
    def __init__(self, d_model):
        super(Encoder, self).__init__()
        # Encoder를 통과한 값의 크기를 조절하기위해 Dense를 사용한다.
        self.fc = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        x = self.fc(inputs)  # 위에서 추출한 image feature가 입력으로 사용됩니다.
        x = tf.nn.relu(x)  # 활성화 함수로 relu 사용
        return x


# ### Transformer Decoder layer 정의
# 3개의 서브 레이어를 가지는 하나의 Decoder layer를 정의합니다.

# In[17]:


class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate):
        super(Decoder_Layer, self).__init__()

        # 위의 코드에서 선언한 MultiHeadAttention 클래스를 사용합니다.
        # 첫 번째 sub layer의 MultiHeadAttention 선언
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # 두 번째 sub layer의 MultiHeadAttention 선언
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        # 입력과 출력의 크기를 같게 하기 위해 position_wise_feed_forward_network 선언
        self.ffn = position_wise_feed_forward_network(d_model, dff)

        '''
        층 정규화를 선언하여 gradient가 폭주하거나 소실되는 문제를 완화시켜 
        gradient가 안정적인 값을 가지게 도와줍니다.
        
        각각의 sub layer마다 사용합니다.
        '''
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # dropout을 사용해 과적합 문제를 해결합니다.
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        
        # 첫 번째 서브 레이어
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, train_y_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # 잔차 연결과 층 정규화
        # out1 = (batch_size, train_y_seq_len, d_model)

        # 두 번째 서브 레이어
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, train_y_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # 잔차 연결과 층 정규화 
        # out2 = (batch_size, train_y_seq_len, d_model)
        
        # 세 번째 서브 레이어
        ffn_output = self.ffn(out2)  # (batch_size, train_y_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # 잔차 연결과 층 정규화 
        # out3 = (batch_size, train_y_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# ### Model의 Decoder 정의
# 여러 Decoder layer를 가지는 Transformer 구조의 Decoder를 정의합니다.

# In[18]:


class Decoder(tf.keras.Model):
    def __init__(self, seq_max_num, d_model, num_heads, dff, num_layers, rate):
        super(Decoder, self).__init__()
        self.seq_max_num = seq_max_num  # 가장 긴 문장 길이
        self.d_model = d_model  # 현 모델의 인코더와 디코더에서의 정해진 입력과 출력의 크기
        self.num_layers = num_layers  # Decoder layer의 개수

        # Decoder의 input을 embedding 후 positional_encoding
        self.embedding = tf.keras.layers.Embedding(self.seq_max_num, self.d_model)
        self.pos_encoding = positional_encoding(self.seq_max_num, d_model)
        
        # Decoder layer 선언
        self.dec_layers = [Decoder_Layer(num_heads, self.d_model, dff, rate)
                           for _ in range(num_layers)]

    def call(self, inputs, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        # 가장 긴 SMILES 식의 길이
        seq_len = tf.shape(inputs)[1]
    
        # SMILES식 embedding + position_encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # Decoder layer를 num_layers 개수만큼 쌓는다.
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights


# ### Transformer 정의

# In[19]:


class Transformer(tf.keras.Model):
    def __init__(self, seq_max_num, d_model, num_heads, dff, num_layers, dropout_rate, vocab_size):
        super(Transformer, self).__init__()

        # Image feature를 입력으로 하는 Encoder 정의
        self.encoder = Encoder(d_model)

        # Transformer 구조의 Decoder 정의
        self.decoder = Decoder(seq_max_num=seq_max_num, d_model=d_model, num_heads=num_heads,
                               dff=dff, num_layers=num_layers, rate=dropout_rate)

        # 다중 클래스 분류 문제를 풀수 있도록, dic의 단어 개수 만큼의 뉴런을 가지는 layer 정의
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, tar, training, look_ahead_mask, padding_mask):
        # Encoder 생성
        enc_output = self.encoder(inputs)
        # Decoder 생성
        dec_output, attention_weights = self.decoder(inputs=tar, enc_output=enc_output, training=training,
                                                     look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        # 다중 클래스 분류 문제를 풀 수 있도록, Dense layer 추가
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


# ### Learning rate 설정
# 
# 처음에는 학습이 잘 되지 않은 상태이므로 learning rate를 빠르게 증가시켜 weights의 변화를 크게 주고, 
# 
# warmup_step이 지나면 learning rate를 천천히 감소시켜 weights 변화의 크기를 줄여나갑니다.

# In[20]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
      Learning Rate를 지정
    warmup_step까지는 linear하게 Learning Rate를 증가시키고,
    이후에는 step의 inverse square root에 비례하도록 감소시킵니다.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model  # 현 모델의 인코더와 디코더에서의 정해진 입력과 출력의 크기
        self.d_model = tf.cast(self.d_model, tf.float32)  # float 형으로 casting

        self.warmup_steps = warmup_steps  # warmup_steps 지정

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)  # 제곱근의 역수를 구함
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# ### Transformer model 생성

# In[21]:


# SMILES 변환을 위한 Model 생성
transformer = Transformer(seq_max_num=seq_max_num, d_model=d_model, num_heads=num_heads,
                          dff=dff, num_layers=num_layers, dropout_rate=dropout_rate, vocab_size=vocab_size)
# learning rate 지정
learning_rate = CustomSchedule(d_model)

# 아래 코드는 learning rate 샘플로, 학습률 변화를 시각화 하기 위한 코드이며 학습에 사용하지 않습니다.
sample_learning_rate = CustomSchedule(d_model=512)
plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


# ### 손실 함수 정의하기
# 
# optimizer로는 Adam, Loss Function은 SparseCategoricalcrossentropy를 사용하였습니다.

# In[22]:


# optimizer는 Adam을 사용
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
# 손실 함수는 Sparse Categorical Crossentropy를 사용
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    """
       loss_function을 정의합니다.
    1. 정답 데이터에서 0을 찾아 bool형식으로 반환합니다. ex) [False False  True  True]
    2. 정답 데이터와 예측한 값을 비교하여 loss를 구합니다. ex) [0.6 0.2 0.1 0.1]
    3. logical_not을 사용해 값을 반전시킵니다. ex) [ True  True False False]
    4. mask 값의 type을 loss_의 type으로 casting 합니다. ex) [1. 1. 0. 0.]
    5. loss_ 값과 mask 값을 곱한 후 loss_값을 구합니다. ex) [0.6 0.2 0.  0. ]
    6. loss_ / mask 를 수행하여 loss를 반환합니다. ex) loss = 0.8 / 2.0
    """
    
    # 1. -> 2.
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # 3.
    loss_ = loss_object(real, pred)

    # 4.
    mask = tf.cast(mask, dtype=loss_.dtype)
    # 5.
    loss_ *= mask

    # 6.
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
print("create model")


# ### CheckPoint 지정

# In[23]:


checkpoint_path = ".\\checkpoint"  # 체크포인터 경로를 설정합니다.
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=25)  # 최대 25개 까지 저장

# 저장된 체크포인터가 있다면 load 후 이어서 training 합니다.
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    print("start_epoch :", start_epoch)
    # checkpoint 복원
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("checkpoint load :", ckpt_manager.latest_checkpoint)


# ### Train step 정의

# In[24]:


@tf.function
def train_step(inp, tar):
    
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    # mask 생성
    combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    """
    combined_mask는 Decoder_layer의 첫 번째 sub_layer에서 사용됩니다.
    
    dec_padding_mask는 아래 논문의 Transformer Model에서는 사용되지만, 
    제안하는 Model에선 두 번째 sub_layer에 image_feature가 사용되므로 padding_mask를 사용하지 않습니다.
    
    Transformer 논문 출처 : http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    """
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     combined_mask,
                                     None)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # loss와 accuracy 기록
    train_loss(loss)
    train_accuracy(tar_real, predictions)


# ### Train 시작

# In[25]:


EPOCHS = 50
print("start train")

for epoch in range(EPOCHS):
    start = time.time()  # epoch당 소요 시간을 체크

    # loss와 accuracy 계산을 위한 metric 사용
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):

        train_step(inp, tar)

        if batch % 200 == 0:
            print('Epoch {} Batch {} Loss {:.6f} Accuracy {:.6f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    # checkpoint 저장
    ckpt_manager.save()

    # Epoch 당 Loss와 Accuracy 출력
    print('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    # Epoch 당 소요 시간 출력
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


# ## 4. TestSet 예측

# ### TestSet 예측을 위한 image load

# In[26]:


test_img_file_path = []  # Test에 사용할 이미지의 경로가 들어갈 리스트
test_img_PATH = '.\\test\\'  # Test에 사용할 이미지가 있는 폴더의 경로

'''sample_submission 파일 경로
sample_submission.SMILES에 Test 결과값을 채웁니다.'''
test_csv = pd.read_csv('.\\sample_submission.csv')  
print(test_csv)

for i in test_csv.file_name:
    test_img_file_path.append(test_img_PATH + i)


# ### 빠른 test를 위해 InceptionResNetV2에서 추출한 feature를 로컬에 저장
# 추출된 (8x8x1536)의 feature를 (64x1536)로 reshape하여 저장합니다

# In[27]:


encode_train = sorted(set(test_img_file_path))

# 데이터를 불러옵니다.
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# dataset의 각 멤버에 사용자 지정 함수(load_image)를 적용합니다. 
image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

num = 0
for img, path in image_dataset:
    # image_feature를 추출합니다.
    batch_features = image_features_extract_model(img)
    # (8, 8, 1536)의 feature를 (64, 1536)으로 reshape합니다.
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
        # test_1.png 그림을 test_1.png.npy 로 저장합니다.
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())
        num += 1
        if num % 1000 == 0:
            print(num)
print("save")


# ### TestSet 정의 함수

# In[28]:


def map_func_pred(img_name, decoder_init_sos):
    """npy 데이터를 읽어옵니다."""
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, decoder_init_sos


# ### 예측 데이터셋 준비

# In[29]:


# predict를 위해 문장의 시작을 알리는 심볼 '<'을 생성
decoder_init_sos = [tokenizer.word_index['<']]
# 데이터 1개마다 심볼 '<'을 1대1 대응 시키기 위해 실행
decoder_init_sos = tf.expand_dims(decoder_init_sos * len(test_img_file_path), 1)  

# 데이터를 불러옵니다.
test_dataset = tf.data.Dataset.from_tensor_slices((test_img_file_path, decoder_init_sos))
# dataset의 각 멤버에 사용자 지정 함수(map_func_pred)를 적용합니다. 
test_dataset = test_dataset.map(lambda item1, item2: tf.numpy_function(map_func_pred, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
# shuffle()을 사용하여 설정된 epoch 마다 dataset 을 섞은 후, batch size 지정
test_dataset = test_dataset.batch(BATCH_SIZE)
# 모델이 s스텝 훈련을 실행하는 동안 입력 파이프라인은 s+1스텝의 데이터를 읽어옵니다.
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ### Test step 정의

# In[30]:


def test_step(inp, decoder_init_sos, test_num):
    # 문장의 시작을 알리는 심볼 '<'
    decoder_input = decoder_init_sos

    # 가장 긴 문장의 길이만큼 반복합니다.
    for i in range(seq_max_num):
        combined_mask, dec_padding_mask = create_masks(inp, decoder_input)
        """
        Test에서도 dec_padding_mask는 사용되지 않습니다.
        """
        predictions, _ = transformer(inp, decoder_input,
                                     False,
                                     combined_mask,
                                     None)
        predictions = predictions[:, -1:, :]

        # test_num == 1 이라면, 가장 높은 확률의 값으로 예측합니다.
        if test_num == 1:
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # test_num == 1이 아니라면, 비교적 높은 확률의 값으로 다시 예측합니다.
        else:
            predictions = tf.reshape(predictions, shape=[predictions.shape[0], predictions.shape[1] * predictions.shape[2]])
            predicted_id = tf.cast(tf.random.categorical(predictions, 1), tf.int32)

        # 이전의 예측 값 뒤에 현재 예측값을 이어붙입니다.
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
    return decoder_input


# ### Test 시작

# In[31]:


print("start test")
test_start_time = time.time()  # TestSet 전체를 predict 할 때 소요된 시간 체크
result = []
for (batch, (inp, init_sos)) in enumerate(test_dataset):
    start_time = time.time()  # batch_size 만큼 predict 할 때 소요된 시간 체크

    '''
    가장 높은 확률로 예측합니다.
    
    test_step의 return값의 shape = (batch_size, seq_max_num+1)이며 
    batch_size개의 예측값을 extend를 사용하여 result에 추가함으로써 result를 1차원 배열로 사용하였습니다.
    '''
    result.extend(test_step(inp, init_sos, 1))

    if batch % 1 == 0 and batch != 0:
        print("testImg_num :", batch * BATCH_SIZE)
        print("Time taken for batch: {} secs\n".format((time.time() - start_time) * 1))
print(len(result))
print("Test time required : {} secs\n".format(time.time() - test_start_time))


# ### SMILES로 변환

# In[32]:


# TestSet을 Predict한 결과값들을 문자열로 변환합니다.
preds = []
for i in range(len(result)):
    # index_word를 사용해 숫자값을 단어로 변경
    # ex) <Cc1ccc(C(=O)N2CCc3ccccc32)cc1N>>>N>>>C)C>C2>>>C>C)C>>C>>>C>>N)=O>N>C>C)C
    pred = ''.join([tokenizer.index_word[i] for i in result[i].numpy()]) 
    # 첫 '>'를 기준으로 '>'의 앞의 단어들만 저장
    # ex) <Cc1ccc(C(=O)N2CCc3ccccc32)cc1N
    pred = pred.split('>')[0]
    # 문장의 시작을 알리는 '<'를 제외하고 저장
    # ex) Cc1ccc(C(=O)N2CCc3ccccc32)cc1N
    preds.append(pred[1:])  # '<'와 '>'가 제외된 문자열이 저장됩니다.


# ### SMILES 규칙을 만족하지 않은 결과 재예측

# In[ ]:


# SMILES 규칙을 만족하지 않은 값들의 index를 error_idx_ 에 추가합니다.
error_idx = []
for i, pred in enumerate(preds):
    m = Chem.MolFromSmiles(pred)  # pred가 SMILES 형식을 만족하는지 검사합니다.
    if m is None:  # SMILES 형식을 만족하지 않으면 m이 None입니다.
        error_idx.append(i)  # 만족하지 않는 값은 값의 index를 error_idx에 추가합니다.
error_idx = np.array(error_idx)
error_idx_ = error_idx.copy()

drop_error = []
# SMILES 식을 만족하지 않는 값들을 re_predict 하기 위해 while문을 사용합니다.
while True:
    error_idx_dict = {}
    for i, e in enumerate(error_idx_):
        error_idx_dict[i] = e

    # re_predict를 위한 DataSet을 만들기 위해 SMILES식을 만족하지 않는 값들의 image 주소를 가져옵니다.
    re_predict_img_name = np.array(test_img_file_path)[error_idx_]

    
    # re_predict를 위한 Dataset을 만들기 위해 문장의 시작을 알리는 '<' 생성
    re_predict_decoder_init_sos = [tokenizer.word_index['<']]
    # 데이터 1개마다 심볼 '<'을 1대1 대응 시키기 위해 실행
    re_predict_decoder_init_sos = tf.expand_dims(re_predict_decoder_init_sos * len(re_predict_img_name), 1)

    
    # re_predict를 위한 DataSet 생성
    re_predict_dataset = tf.data.Dataset.from_tensor_slices((re_predict_img_name, re_predict_decoder_init_sos))
    # dataset의 각 멤버에 사용자 지정 함수(map_func_pred)를 적용합니다. 
    re_predict_dataset = re_predict_dataset.map(lambda item1, item2: tf.numpy_function(map_func_pred, [item1, item2], [tf.float32, tf.int32]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # shuffle()을 사용하여 설정된 epoch 마다 dataset 을 섞은 후, batch size 지정
    re_predict_dataset = re_predict_dataset.batch(BATCH_SIZE)
    # 모델이 s스텝 훈련을 실행하는 동안 입력 파이프라인은 s+1스텝의 데이터를 읽어옵니다.
    re_predict_dataset = re_predict_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    
    # 비교적 높은 확률로 re_predict 후 test_result에 저장합니다.
    test_result_ = []
    for (batch, (inp, init_sos)) in enumerate(re_predict_dataset):
        '''
        test_step의 return값의 shape = (batch_size, seq_max_num+1)이며 
        batch_size개의 예측값을 extend를 사용하여 result에 추가함으로써 result를 1차원 배열로 사용하였습니다.
        '''
        test_result_.extend(test_step(inp, init_sos, 2))
    test_result_ = np.array(test_result_)

    # re_predict한 값들을 문자열로 변환합니다.
    preds_ = []
    for rid in range(len(test_result_)):
        # index_word를 사용해 숫자값을 단어로 변경
        pred = ''.join([tokenizer.index_word[i] for i in test_result_[rid]])
        # 첫 '>'를 기준으로 '>'의 앞의 단어들만 저장
        pred = pred.split('>')[0]
        # 문장의 시작을 알리는 '<'를 제외하고 저장
        preds_.append(pred[1:])   # '<'와 '>'가 제외된 문자열이 저장됩니다.

    # SMILES식을 만족하는 값들을 error_idx_ 리스트에서 지웁니다.
    for i, pred in enumerate(preds_):
        m = Chem.MolFromSmiles(pred)  # pred가 SMILES 형식을 만족하는지 검사합니다.
        if m is not None:  # SMILES 형식을 만족하지 않으면 m이 None입니다.
            preds[error_idx_dict[i]] = pred
            drop_idx = np.where(error_idx == error_idx_dict[i])[0]
            drop_error.append(drop_idx[0])
    # re_predict 결과 SMILES 형식을 만족한 값을 error_idx_ 에서 삭제합니다.
    error_idx_ = np.delete(error_idx, drop_error)  

    # re_predict 결과 SMILES 형식을 만족한 값들의 개수 / 첫 predict 결과 SMILES 형식이 아닌 값들의 개수
    print(len(list(drop_error)), '/', error_idx.shape[0])

    # SMILES 형식이 아닌 값들의 개수가 10개 미만이면 re_predict를 멈춥니다.
    if error_idx.shape[0] - len(list(drop_error)) < 10:
        break


# ## 5. 제출

# In[ ]:


test_csv['SMILES'] = preds
# result.csv 파일로 저장합니다.
test_csv.to_csv('.\\result.csv', index=False)

