import re   #정규식
import os   #디렉토리 관리
import io
import random   #학습 데이터 섞기
import tarfile #gz 압축 풀기


import numpy as np   #행렬 연산
import pandas as pd   #데이터프레임
import tensorflow as tf   #신경망


# from eunjeon  import Mecab
# from konlpy.tag import Mecab
# import sentencepiece as spm   #SentencePiece


from tqdm.notebook import tqdm   #학습과정 시각화
from tqdm import tqdm_notebook   #학습과정 시각화


import seaborn   #데이터 시각화
import matplotlib as mpl   #폰트
import matplotlib.pyplot as plt   #데이터 시각화
import matplotlib.font_manager as fm   #폰트
import sqlite3  

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager.findfont(font)

import pickle
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


mecab_enc_tokenizer_path = 'C:/Users/youngdong/Desktop/Text-to-Color-main/models/mecab_enc_tokenizer.pickle'
mecab_dec_tokenizer_path = 'C:/Users/youngdong/Desktop/Text-to-Color-main/models/mecab_dec_tokenizer.pickle'
mecab_enc_tensor_path = 'C:/Users/youngdong/Desktop/Text-to-Color-main/models/mecab_enc_tensor.pickle'
mecab_dec_tensor_path = 'C:/Users/youngdong/Desktop/Text-to-Color-main/models/mecab_dec_tensor.pickle'
mecab_dec_ = 'C:/Users/youngdong/Desktop/Text-to-Color-main/models/model_210519.pickle'
with open(mecab_enc_tokenizer_path,"rb") as fr:
    mecab_enc_tokenizer = pickle.load(fr)

with open(mecab_dec_tokenizer_path,"rb") as fr:
    mecab_dec_tokenizer = pickle.load(fr)

with open(mecab_enc_tensor_path,"rb") as fr:
    mecab_enc_tensor = pickle.load(fr)

with open(mecab_dec_tensor_path,"rb") as fr:
    mecab_dec_tensor = pickle.load(fr)


#데이터 길이 시각화 함수=========================
def show_sentence_length(sentence_num, title, range_=[0, 500]):
    plt.figure(figsize=(13, 5))
    plt.suptitle(title, fontsize=14)
    
    plt.subplot(1, 2, 1)
    plt.hist(sentence_num, bins=range_[1], range=range_, facecolor='b', label='train')
    plt.xlabel('Number of question')
    plt.ylabel('Count of question')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(sentence_num, labels=['token counts'], showmeans=True)

    plt.show()
    
    print("< Sentence Info >".center(100, "="))
    print(f"길이 최대:    {np.max(sentence_num):4d}")
    print(f"길이 최소:    {np.min(sentence_num):4d}")
    print(f"길이 평균:    {np.mean(sentence_num):7.3f}")
    print(f"길이 표준편차: {np.std(sentence_num):7.3f}", end="\n\n")
    
    percentile25 = np.percentile(sentence_num, 25)
    percentile50 = np.percentile(sentence_num, 50)
    percentile75 = np.percentile(sentence_num, 75)
    percentileIQR = percentile75 - percentile25
    percentileMAX = percentile75 + percentileIQR * 1.5
    
    print(f" 25/100분위:  {percentile25:7.3f}")
    print(f" 50/100분위:  {percentile50:7.3f}")
    print(f" 75/100분위:  {percentile75:7.3f}")
    print(f" MAX/100분위: {percentileMAX:7.3f}")
    print(f" IQR: {percentileIQR:7.3f}")
    print("=" * 100)
#End===========================================

def filt_sentence_length(df, col, sentence_len):
    df = df.copy()
    df["len"] = df[col].apply(lambda x: len(x.split()))
    df = df.loc[df["len"] < sentence_len]
    df.drop(["len"], axis="columns", inplace=True)
    return df

#전처리 함수===========================
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([0-9?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Zㄱ-ㅎ가-힣0-9?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence
#End===================================

#sos 및 eos 토큰 추가==================
def append_os_token(sentence):
    sentence = "<sos> " + sentence + " <eos>"
    return sentence
#End===================================

def mecab_tokenize(corpus, vocab_size, maxlen, encoder_TF=True):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        oov_token="<UNK>",
        num_words=vocab_size
    )
    corpus_input = corpus
    
    if encoder_TF:   #encoder data는 '한국어'이므로 mecab 형태소 분석
        m = Mecab()
    
        corpus_input = []
        for sentence in corpus:
            corpus_input.append(m.morphs(sentence))
    
    tokenizer.fit_on_texts(corpus_input)
    
    if vocab_size is not None:
        words_frequency = [w for w,c in tokenizer.word_index.items() if c >= vocab_size + 1]
        for w in words_frequency:
            del tokenizer.word_index[w]
            del tokenizer.word_counts[w]
    
    tensor = tokenizer.texts_to_sequences(corpus_input)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor,
        padding='post',
        maxlen=maxlen
    )
    return tensor, tokenizer

def wordNumByFreq(tokenizer, freq_num):
    sorted_freq = sorted(tokenizer.word_counts.items(), key=lambda x: x[1])
    for idx, (_, freq) in enumerate(sorted_freq):
        if freq > freq_num: break;
    return idx

#Positional Encoding==================================================
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, int(i)/d_model)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]
    
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    
    return sinusoid_table
#End==================================================================


#MultiHeadAttention====================================================
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
            
        self.depth = d_model // self.num_heads
            
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
            
        self.linear = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = tf.cast(K.shape[-1], tf.float32)
        QK = tf.matmul(Q, K, transpose_b=True)

        scaled_qk = QK / tf.math.sqrt(d_k)

        if mask is not None: scaled_qk += (mask * -1e9)  

        attentions = tf.nn.softmax(scaled_qk, axis=-1)
        out = tf.matmul(attentions, V)

        return out, attentions
            

    def split_heads(self, x):
        batch_size = x.shape[0]
        split_x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        split_x = tf.transpose(split_x, perm=[0, 2, 1, 3])

        return split_x

    def combine_heads(self, x):
        batch_size = x.shape[0]
        combined_x = tf.transpose(x, perm=[0, 2, 1, 3])
        combined_x = tf.reshape(combined_x, (batch_size, -1, self.d_model))

        return combined_x

        
    def call(self, Q, K, V, mask):
        WQ = self.W_q(Q)
        WK = self.W_k(K)
        WV = self.W_v(V)
        
        WQ_splits = self.split_heads(WQ)
        WK_splits = self.split_heads(WK)
        WV_splits = self.split_heads(WV)
            
        out, attention_weights = self.scaled_dot_product_attention(
            WQ_splits, WK_splits, WV_splits, mask
        )
        out = self.combine_heads(out)
        out = self.linear(out)
                
        return out, attention_weights
#End==================================================================

    
#Position-wise Feed-Forward Network===================================
class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        out = self.w_1(x)
        out = self.w_2(out)
            
        return out
#End==================================================================


#Mask 레이어==========================================================
def generate_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def generate_causality_mask(src_len, tgt_len):
    mask = 1 - np.cumsum(np.eye(src_len, tgt_len), 0)
    return tf.cast(mask, tf.float32)

def generate_masks(src, tgt):
    enc_mask = generate_padding_mask(src)
    dec_mask = generate_padding_mask(tgt)

    dec_enc_causality_mask = generate_causality_mask(tgt.shape[1], src.shape[1])
    dec_enc_mask = tf.maximum(enc_mask, dec_enc_causality_mask)

    dec_causality_mask = generate_causality_mask(tgt.shape[1], tgt.shape[1])
    dec_mask = tf.maximum(dec_mask, dec_causality_mask)

    return enc_mask, dec_enc_mask, dec_mask
#End==================================================================

#Encoder 레이어=======================================================
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):
        residual = x
        out = self.norm_1(x)
        out, enc_attn = self.enc_self_attn(out, out, out, mask)
        out = self.dropout(out)
        out += residual
        
        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual
        
        return out, enc_attn
#End==================================================================


#Decoder 레이어=======================================================
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, enc_out, causality_mask, padding_mask):
        residual = x
        out = self.norm_1(x)
        out, dec_attn = self.dec_self_attn(out, out, out, padding_mask)
        out = self.dropout(out)
        out += residual

        residual = out
        out = self.norm_2(out)
        out, dec_enc_attn = self.enc_dec_attn(out, enc_out, enc_out, causality_mask)
        out = self.dropout(out)
        out += residual
       
        residual = out
        out = self.norm_3(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual

        return out, dec_attn, dec_enc_attn
#End==================================================================

#Encoder==============================================================
class Encoder(tf.keras.Model):
    def __init__(self,
                 n_layers,
                 d_model,
                 n_heads,
                 d_ff,
                 dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.enc_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) 
                        for _ in range(n_layers)]
        
    def call(self, x, mask):
        out = x
    
        enc_attns = list()
        for i in range(self.n_layers):
            out, enc_attn = self.enc_layers[i](out, mask)
            enc_attns.append(enc_attn)
        
        return out, enc_attns
#End==================================================================


#Decoder==============================================================
class Decoder(tf.keras.Model):
    def __init__(self,
                 n_layers,
                 d_model,
                 n_heads,
                 d_ff,
                 dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.dec_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) 
                            for _ in range(n_layers)]
                            
                            
    def call(self, x, enc_out, causality_mask, padding_mask):
        out = x
    
        dec_attns = list()
        dec_enc_attns = list()
        for i in range(self.n_layers):
            out, dec_attn, dec_enc_attn = \
            self.dec_layers[i](out, enc_out, causality_mask, padding_mask)

            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)

        return out, dec_attns, dec_enc_attns
#End==================================================================

class Transformer(tf.keras.Model):
    def __init__(self,
                    n_layers,
                    d_model,
                    n_heads,
                    d_ff,
                    src_vocab_size,
                    tgt_vocab_size,
                    pos_len,
                    dropout=0.2,
                    shared=True):
        super(Transformer, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)

        self.enc_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.dec_emb = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = positional_encoding(pos_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.fc = tf.keras.layers.Dense(tgt_vocab_size)

        self.shared = shared

        if shared: self.fc.set_weights(tf.transpose(self.dec_emb.weights))

            
    def embedding(self, emb, x):
        seq_len = x.shape[1]
        out = emb(x)

        if self.shared: out *= tf.math.sqrt(self.d_model)

        out += self.pos_encoding[np.newaxis, ...][:, :seq_len, :]
        out = self.dropout(out)

        return out

        
    def call(self, enc_in, dec_in, enc_mask, causality_mask, dec_mask):
        enc_in = self.embedding(self.enc_emb, enc_in)
        dec_in = self.embedding(self.dec_emb, dec_in)

        enc_out, enc_attns = self.encoder(enc_in, enc_mask)
        
        dec_out, dec_attns, dec_enc_attns = \
        self.decoder(dec_in, enc_out, causality_mask, dec_mask)
        
        logits = self.fc(dec_out)
        
        return logits, enc_attns, dec_attns, dec_enc_attns
#LearningRateScheduler=====================
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)
#End=======================================


#손실 함수=================================  
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)    
#End=======================================

    
learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# Train Step 함수======================================

@tf.function()
def train_step(src, tgt, model, optimizer):
    gold = tgt[:, 1:]
        
    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt)

    with tf.GradientTape() as tape:
        predictions, enc_attns, dec_attns, dec_enc_attns = \
        model(src, tgt, enc_mask, dec_enc_mask, dec_mask)
        loss = loss_function(gold, predictions[:, :-1])

    # 최종적으로 optimizer.apply_gradients()가 사용됩니다. 
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, enc_attns, dec_attns, dec_enc_attns
#End===================================================

# Attention 시각화 함수================================
def visualize_attention(src, tgt, enc_attns, dec_attns, dec_enc_attns):
    def draw(data, ax, x="auto", y="auto"):
        seaborn.heatmap(
            data, 
            square=True,
            vmin=0.0, vmax=1.0, 
            cbar=False, ax=ax,
            xticklabels=x, yticklabels=y
        )
        
    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print(f"< Encoder Layer(=Self Attention) {layer + 1} >".center(100, "="))
        for h in range(4):
            draw(enc_attns[layer][0, h, :len(src), :len(src)], axs[h], src, src)
        plt.show()
        print("=" * 100, end="\n\n\n")
        
    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print(f"< Decoder Self Layer {layer + 1} >".center(100, "="))
        for h in range(4):
            draw(dec_attns[layer][0, h, :len(tgt), :len(tgt)], axs[h], tgt, tgt)
        plt.show()
        print("=" * 100, end="\n\n\n")

        print(f"< Decoder Layer(Context Vector By Decoder Input) {layer + 1} >".center(100, "="))
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(dec_enc_attns[layer][0, h, :len(tgt), :len(src)], axs[h], src, tgt)
        plt.show()
        print("=" * 100, end="\n\n\n")
#End===================================================


#번역 함수=============================================
def evaluate(
    sentence,
    model,
    src_tokenizer, tgt_tokenizer,
    enc_maxlen, dec_maxlen,
    tokenizer_type
):
    #tokenizer_type = True: Mecab, False: SentencePiece
    
    def idx_to_text(idx, tokenizer_type):
        if tokenizer_type:
            return tgt_tokenizer.sequences_to_texts([idx])
        else:
            return tgt_tokenizer.decode_ids(idx)
        
    sentence = preprocess_sentence(sentence)
    
    if tokenizer_type:
        m = Mecab()
        sentence = m.morphs(sentence)
        pieces = sentence
        _input = src_tokenizer.texts_to_sequences([sentence])
        
        sos_idx = tgt_tokenizer.word_index['<sos>']
        eos_idx = tgt_tokenizer.word_index['<eos>']
    else:
        pieces = src_tokenizer.encode_as_pieces(sentence)
        _input = [src_tokenizer.encode_as_ids(sentence)]
        
        sos_idx = tgt_tokenizer.bos_id()
        eos_idx = tgt_tokenizer.eos_id()
        
    _input = tf.keras.preprocessing.sequence.pad_sequences(
        _input,
        maxlen=enc_maxlen,
        padding='post'
    )
    
    ids = []
    output = tf.expand_dims([sos_idx], 0)
    
    for i in range(dec_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = \
        generate_masks(_input, output)
        
        predictions, enc_attns, dec_attns, dec_enc_attns =\
        model(_input, output, enc_padding_mask, combined_mask, dec_padding_mask)
        
        predicted_id = tf.argmax(
            tf.math.softmax(predictions, axis=-1)[0, -1]
        ).numpy().item()
        
        if predicted_id == eos_idx:
            result = idx_to_text(ids, tokenizer_type)
            return pieces, result, enc_attns, dec_attns, dec_enc_attns
        
        ids.append(predicted_id)
        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)
    result = idx_to_text(ids, tokenizer_type)
    return pieces, result, enc_attns, dec_attns, dec_enc_attns
#End===================================================



#번역 함수2============================================
def translate(
    sentence,
    model,
    src_tokenizer, tgt_tokenizer,
    enc_maxlen, dec_maxlen,
    plot_attention=False,
    tokenizer_type=True
):
    pieces, result, enc_attns, dec_attns, dec_enc_attns = \
    evaluate(
        sentence,
        model,
        src_tokenizer, tgt_tokenizer,
        enc_maxlen, dec_maxlen,
        tokenizer_type=tokenizer_type
    )
    if tokenizer_type:
        result = " ".join(result) 
    # if plot_attention:
    #     visualize_attention(pieces, result.split(), enc_attns, dec_attns, dec_enc_attns)
    else:
        print("Korean Sentence:".rjust(18), sentence)
        print("English Sentence:".rjust(18), result, end="\n\n")

#End===================================================
transformer = Transformer(
    n_layers=2,
    d_model=256,
    n_heads=8,
    d_ff=128,
    dropout=0.2,
    pos_len=200,
    shared=True,
    src_vocab_size=1 , tgt_vocab_size = 1
)
checkpoint_path =  "C:/Users/youngdong/Desktop/Text-to-Color-main/models/checkpoint"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

# 체크포인트가 있으면 최신 체크포인트를 복원합니다.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('최신 체크포인트가 복원되었습니다!!')


BATCH_SIZE = 64
EPOCHS = 10

examples = [
    "오바마는 대통령이다.", "시민들은 도시 속에 산다.",
    "커피는 필요 없다.", "일곱 명의 사망자가 발생했다.",
    "나는 고양이다.", "아버지가 방에 들어가신다.",
    "아버지 가방에 들어가신다.", "오늘은 크리스마스다."
]

# def fit(enc_train, dec_train, enc_tokenizer, dec_tokenizer, tokenizer_type=True):
#     enc_maxlen, dec_maxlen = enc_train.shape[-1], dec_train.shape[-1]
    
#     for epoch in range(EPOCHS):
#         total_loss = 0

#         idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
#         random.shuffle(idx_list)
#         t = tqdm_notebook(idx_list)

#         for (batch, idx) in enumerate(t):
#             batch_loss, enc_attns, dec_attns, dec_enc_attns = \
#             train_step(
#                 enc_train[idx:idx+BATCH_SIZE],
#                 dec_train[idx:idx+BATCH_SIZE],
#                 transformer,
#                 optimizer
#             )

#             total_loss += batch_loss

#             t.set_description_str('Epoch %2d' % (epoch + 1))
#             t.set_postfix_str('Loss %.3f' % (total_loss.numpy() / (batch + 1)))
        
#         if (epoch+1) % 5 == 0:
#             print(f"< EPOCH {epoch} >".center(100, "=")) 
#             for example in examples: 
#                 translate(
#                     example,
#                     transformer,
#                     enc_tokenizer, dec_tokenizer,
#                     enc_maxlen, dec_maxlen,
#                     tokenizer_type=tokenizer_type,
#                     plot_attention=False
#                 )
#             print("=" * 100, end="\n\n\n")
        
#         if (epoch+1) == EPOCHS:
#             translate(
#                 examples[0],
#                 transformer,
#                 enc_tokenizer, dec_tokenizer,
#                 enc_maxlen, dec_maxlen,
#                 tokenizer_type=tokenizer_type,
#                 plot_attention=True
#             )
#         # if (epoch + 1) % 5 == 0:
#         #   ckpt_save_path = ckpt_manager.save()
#         #   print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
#         #                                                  ckpt_save_path))

translate('책읽으러 가야해요',  transformer,  mecab_enc_tokenizer, mecab_dec_tokenizer,  mecab_enc_tensor.shape[-1], mecab_dec_tensor.shape[-1], plot_attention=False,  tokenizer_type=True)
