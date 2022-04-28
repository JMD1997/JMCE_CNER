#!/usr/bin/env python
# coding: utf-8

# # 加载词表

# In[1]:


import jieba
jieba.load_userdict("new_userdict.txt")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# # 读取数据集

# In[2]:


from keras.layers import Bidirectional,LSTM

import pandas as pd

tag=['STR', 'BRA', 'ATT', 'CAR']

# tag=["/loc"]

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import  ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
maxlen = 256
epochs = 10
batch_size = 24
bert_layers = 12
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 10  # 必要时扩大CRF层的学习率

# bert配置
config_path = '../chinese_roberta_www_ext/bert_config.json'
checkpoint_path = '../chinese_roberta_www_ext/bert_model.ckpt'
dict_path = '../chinese_roberta_www_ext/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：[(片段1, 标签1), (片段2, 标签2), (片段3, 标签3), ...]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D

# 标注数据

train_data = load_data('数据/train.txt')
valid_data = load_data('数据/val.txt')
test_data = load_data('数据/test.txt')


# # 训练词向量

# In[3]:


sentence=[]
for i in train_data:
    tmpsen=""
    for j in i:
        tmpsen+=j[0]
    sentence.append(list(jieba.cut(tmpsen)))

for i in valid_data:
    tmpsen=""
    for j in i:
        tmpsen+=j[0]
    sentence.append(list(jieba.cut(tmpsen)))

for i in test_data:
    tmpsen=""
    for j in i:
        tmpsen+=j[0]
    sentence.append(list(jieba.cut(tmpsen)))


# In[4]:


dfw2v=pd.read_excel("dataSet.xlsx")


# In[6]:


dfw2v=dfw2v.fillna("")


# In[8]:


for i in tqdm(list(dfw2v["为什么购买"])+list(dfw2v["最满意"])+list(dfw2v["最不满意"])+list(dfw2v["其它评论"])):
    sentence.append(list(jieba.cut(i)))


# In[9]:


import pandas as pd
import gensim
w2v_model = gensim.models.Word2Vec(sentence, size=32, iter=10, min_count=0)
word_vectors = w2v_model.wv

w2v_word2id=dict(zip(word_vectors.index2word,range(3,len(word_vectors.index2word)+3)))

id2w2v_word=dict(zip(w2v_word2id.values(),w2v_word2id.keys()))


# # 建立分词器

# In[10]:


tokenizer = Tokenizer(dict_path, do_lower_case=True)


# # 类别映射

# In[11]:


labels = [i for i in tag]
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


# # 读取之前输出的对应表

# In[12]:


tmpdf=pd.read_csv("newtabel.txt",sep=" ")


# In[13]:


tmpdf


# # 制作部首的映射

# In[14]:


d_rad=dict(zip(list(tmpdf["汉字"]),list(tmpdf["部首"])))
d_rad_values=set(d_rad.values())
d_rad2_index=dict(zip(d_rad_values,range(3,len(d_rad_values)+3)))


# # 制作全拼映射

# In[15]:


d_pin=dict(zip(list(tmpdf["汉字"]),list(tmpdf["五笔"])))
d_pin_values=set(d_pin.values())
d_pin2_index=dict(zip(d_pin_values,range(3,len(d_pin_values)+3)))


# # 制作五笔映射

# In[16]:


d_wubi=dict(zip(list(tmpdf["汉字"]),list(tmpdf["全拼"])))
d_wubi_values=set(d_wubi.values())
d_wubi2_index=dict(zip(d_wubi_values,range(3,len(d_wubi_values)+3)))


# # 保存映射关系

# In[17]:


import joblib
if not os.path.exists("d_rad2_index.joblib"):
     joblib.dump(d_rad2_index,"d_rad2_index.joblib")
else:
     d_rad2_index=joblib.load("d_rad2_index.joblib")
        

if not os.path.exists("d_pin2_index.joblib"):
     joblib.dump(d_pin2_index,"d_pin2_index.joblib")
else:
     d_pin2_index=joblib.load("d_pin2_index.joblib")
        
        

if not os.path.exists("d_wubi2_index.joblib"):
     joblib.dump(d_wubi2_index,"d_wubi2_index.joblib")
else:
     d_wubi2_index=joblib.load("d_wubi2_index.joblib")


# In[18]:


train_generator = DataGenerator(train_data, batch_size)


# # 词和bert的字对应的一个函数

# In[19]:


def get_w2v_index(token_ids,tmp):
    vis=True
    lastindex=1
    wordvec=[1]
    for j in list(jieba.cut(tmp)):
        w_token_ids = tokenizer.encode(j)[0][1:-1]
#         if j=="年":
#             print ("zhuyi:",lastindex)
        sign=find_head_idx(token_ids,w_token_ids,lastindex)
        if sign!=lastindex and sign!=-1 and vis:
             vis=False
             firstword=j
        else:
#             print ("zhuyi:",sign,lastindex)
            if sign==-1 and vis:
                vis=False
                firstword=j
            if sign!=-1 and not vis:
                vis=True
    #             print (sign)
                for i in range(lastindex,sign):
                    try:
                        wordvec.append(w2v_word2id[firstword])
                    except:
                        wordvec.append(3)
#                 print ("len0",len(wordvec),j)
    #             lastindex=sign
            if sign!=-1:
                lastindex=sign+len(w_token_ids)
#                 print(lastindex)
                for i in range(sign,lastindex):
                    try:
                        wordvec.append(w2v_word2id[j])
                    except:
                        wordvec.append(3)
#                 print ("len",len(wordvec),j)
                vis=True
    if sign==-1 and not vis:
            for i in range(lastindex,len(token_ids)-1):
#                     print (j)
                    try:
                         wordvec.append(w2v_word2id[j])
                    except:
                         wordvec.append(3)
    wordvec.append(1)
    return wordvec

def find_head_idx(source,target,lastindex):
    source=source[lastindex:]
    targetlen=len(target)
    for i in range(len(source)):
        if source[i:i+targetlen]==target:
            return i+lastindex
    return -1


# # 数据生成器-喂给模型

# In[20]:


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels,batch_radfea,batch_pinyin,batch_wubi,batch_wordvec = [], [], [],[],[],[],[]
        for is_end, item in self.sample(random):
            token_ids,radfea,pinyin,wubi,labels = [tokenizer._token_start_id],[2],[2],[2],[0]
            tmp=""
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    for i in w_token_ids:

                            try:
                                radfea.append(d_rad2_index[d_rad[tokenizer.id_to_token(i)]])
                            except:
                                radfea.append(1)
#                                 print ("这个偏旁报错{}".format(tokenizer.id_to_token(i)))
                            try:
                                pinyin.append(d_pin2_index[d_pin[tokenizer.id_to_token(i)]])
                            except:
                                pinyin.append(1)
#                                 print ("这个拼音报错{}".format(tokenizer.id_to_token(i)))
                                
                            try:
                                wubi.append(d_wubi2_index[d_wubi[tokenizer.id_to_token(i)]])
                            except:
                                wubi.append(1)
#                                 print ("这个五笔报错{}".format(tokenizer.id_to_token(i)))
#                             break
                if len(token_ids) + len(w_token_ids) < maxlen:
                    tmp+=w
                    token_ids += w_token_ids
                    
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            
            radfea+=[2]
            pinyin+=[2]
            wubi+=[2]
            
            token_ids += [tokenizer._token_end_id]
            if tmp=="":
                continue
            wordvec=get_w2v_index(token_ids,tmp)
            labels += [0]
            segment_ids = [0] * len(token_ids)

#             print (len(pinyin),len(radfea),len(wubi),len(token_ids))
            assert len(pinyin)==len(token_ids)==len(radfea)==len(wubi)==len(wordvec)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            
            batch_radfea.append(radfea)
            batch_pinyin.append(pinyin)
            batch_wubi.append(wubi)
            batch_wordvec.append(wordvec)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_radfea=sequence_padding(batch_radfea)
                batch_pinyin=sequence_padding(batch_pinyin)
                batch_wubi=sequence_padding(batch_wubi)
                batch_wordvec=sequence_padding(batch_wordvec)
                for j in [batch_token_ids, batch_segment_ids,batch_radfea,batch_pinyin,batch_wubi,batch_labels,batch_wordvec]:
                    if np.isnan(j).any():
                        print (j)
                yield [batch_token_ids, batch_segment_ids,batch_radfea,batch_pinyin,batch_wubi,batch_wordvec], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels,batch_radfea,batch_pinyin,batch_wubi,batch_wordvec  = [],[], [], [],[],[],[]
                
                


# In[21]:


# train_generator = data_generator(train_data, batch_size)

# for i in train_data:
#     for j in i:
#         if j[0]==np.nan:
#             print (1)

# np.isnan(next(train_generator.forfit())[0]).any()

# next(train_generator.forfit())


# # 初始化词向量嵌入矩阵

# In[22]:


embedding_matrix = np.zeros((len(id2w2v_word.keys()) + 3, 32), dtype=np.float32)
not_in_model = 0
in_model = 0
embedding_max_value = 0
embedding_min_value = 1
not_words = []

for word, i in w2v_word2id.items():
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.array(w2v_model[word])
    else:
        not_in_model += 1
        not_words.append(word)


# # bert的输出

# In[23]:


"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output


# In[24]:


from keras.layers import *


# # 偏旁层嵌入

# In[25]:


from keras.layers.embeddings import Embedding
def get_rad_embed():
    embed = Embedding(len(d_rad2_index)+4, 32,trainable=True,mask_zero=True)  #定义一个词嵌入层,将句子转化成对应的向量
    input_rad=Input(shape=(None,))
    emb_rad=embed(input_rad)
    outrad=Dense(16)(emb_rad)
    return input_rad,outrad


# # 拼音嵌入

# In[26]:


def get_pinyin_embed():
    embed = Embedding(len(d_pin2_index)+4, 32,trainable=True,mask_zero=True)  #定义一个词嵌入层,将句子转化成对应的向量
    input_pinyin=Input(shape=(None,))
    emb_pinyin=embed(input_pinyin)
    outpinyin=Dense(16)(emb_pinyin)
    return input_pinyin,outpinyin


# # 五笔嵌入

# In[27]:


def get_wubi_embed():
    embed = Embedding(len(d_wubi2_index)+4, 32,trainable=True,mask_zero=True)  #定义一个词嵌入层,将句子转化成对应的向量
    input_wubi=Input(shape=(None,))
    emb_wubi=embed(input_wubi)
    outwubi=Dense(16)(emb_wubi)
    return input_wubi,outwubi


# # 词嵌入

# In[28]:


def get_word_embed():
    embed = Embedding(len(id2w2v_word.keys())+3, 32, weights=[embedding_matrix],trainable=True,mask_zero=True)  #定义一个词嵌入层,将句子转化成对应的向量
    input_word=Input(shape=(None,))
    emb_word=embed(input_word)
#     outwubi=LSTM(32,return_sequences=True)(emb_wubi)
    emb_word=Dense(16)(emb_word)
    return input_word,emb_word


# In[29]:


input_rad,outrad=get_rad_embed()
input_pinyin,outpinyin=get_pinyin_embed()
input_wubi,outwubi=get_wubi_embed()
input_word,emb_word=get_word_embed()


# # 拼接所有的嵌入

# In[30]:


output=Concatenate()([output,outrad,outpinyin,outwubi,emb_word])


# # lstm

# In[31]:


# output=Bidirectional(LSTM(32,return_sequences=True))(output)


# In[32]:


from bert4keras.layers import MultiHeadAttention


# # 注意力机制

# In[33]:


atten=MultiHeadAttention(8,16,128)
output=atten([output,output,output])


# In[34]:


output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)


# In[35]:


# from bert4keras.optimizers import AdaFactor

# learning_rate = 5e-4
# optimizer = AdaFactor(
#     learning_rate=learning_rate, beta1=0.9, min_dim_size_to_factor=10**6
# )


# In[36]:


model = Model(model.input+[input_rad,input_pinyin,input_wubi,input_word], output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


# In[37]:


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        text=text[:maxlen]
        tokens = tokenizer.tokenize(text)
#         wordtype=[0]+get_text_label(text[:254])+[0]
        while len(tokens) > 512:
            tokens.pop(-2)
#             wordtype.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        radfea=[2]
        pinyin=[2]
        wubi=[2]
        for i in token_ids[1:-1]:
                    
                        
                        try:
                            radfea.append(d_rad2_index[d_rad[tokenizer.id_to_token(i)[0]]])
                        except:
                            radfea.append(1)
#                             print ("这个偏旁报错{}".format(tokenizer.id_to_token(i)))
                        try:
                            pinyin.append(d_pin2_index[d_pin[tokenizer.id_to_token(i)[0]]])
                        except:
                            pinyin.append(1)
#                             print ("这个拼音报错{}".format(tokenizer.id_to_token(i)))

                        try:
                            wubi.append(d_wubi2_index[d_wubi[tokenizer.id_to_token(i)[0]]])
                        except:
                            wubi.append(1)
#                             print ("这个wubi报错{}".format(tokenizer.id_to_token(i)))
        radfea.append(2)
        pinyin.append(2)
        wubi.append(2)
        wordvec=get_w2v_index(token_ids,text)
#         print (len(wordtype),len(token_ids),len())
        token_ids, segment_ids,b_radfea,b_pinyin,b_wubi,b_word = to_array([token_ids], [segment_ids],[radfea],[pinyin],[wubi],[wordvec])
#         for j in [token_ids, segment_ids,b_radfea,b_wordtype]:
#             print (j.shape)
        nodes = model.predict([token_ids, segment_ids,b_radfea,b_pinyin,b_wubi,b_word])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
#         print (len(text))
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


# In[38]:


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
#         if epoch>=10:
#             print (epoch)
            if epoch==0 or epoch%1==0:

                trans = K.eval(CRF.trans)
                NER.trans = trans
        #         print(NER.trans)
                f1, precision, recall = evaluate(valid_data)
                # 保存最优
                if f1 >= self.best_val_f1:
                    self.best_val_f1 = f1
                    model.save_weights('./best_model_318atten_new.weights')
                print(
                    'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                    (f1, precision, recall, self.best_val_f1)
                )
                f1, precision, recall = evaluate(test_data)
                print(
                    'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
                    (f1, precision, recall)
                )


# In[39]:


import time
if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    st=time.time()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=30,
        callbacks=[evaluator]
    )
    et=time.time()
    print ("训练花费时间:{}".format(et-st))

else:

    model.load_weights('./best_model_318atten_new.weights')
    NER.trans = K.eval(CRF.trans)


# In[42]:


print ("训练花费时间:{} s".format(et-st))


# In[43]:


model.load_weights('./best_model_318atten_new.weights')
NER.trans = K.eval(CRF.trans)


# In[44]:


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    alld={}
    for i in tag:
        alld[i]=[1e-10, 1e-10, 1e-10]
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        for t in tag:
            tmptag=t
            tmpR=set([tuple(i) for i in R if i[1] ==tmptag])
            tmpT=set([tuple(i) for i in T if i[1] ==tmptag])
            alld[t][0]+=len(tmpR & tmpT)
            alld[t][1]+=len(tmpR)
            alld[t][2]+=len(tmpT)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    allf1=[]
    allpre=[]
    allrecall=[]
    for key in alld.keys():
        print("cate:",key)
        print('valid:  f1: %.5f, precision: %.5f, recall: %.5f\n' %(2*alld[key][0]/(alld[key][1]+alld[key][2]),alld[key][0]/(alld[key][1]),alld[key][0]/(alld[key][2])))
        allf1.append(2*alld[key][0]/(alld[key][1]+alld[key][2]))
        allpre.append(alld[key][0]/(alld[key][1]))
        allrecall.append(alld[key][0]/(alld[key][2]))
    print("mac f1:",sum(allf1)/len(tag))
    print ("mac precision:",sum(allpre)/len(tag))
    print ("mac recall:",sum(allrecall)/len(tag))
#     print("mac f1:",sum(allf1)/len(tag))
    return f1, precision, recall


# In[45]:


f1, precision, recall = evaluate(valid_data)
print(
            'test: mic f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


# In[46]:


f1, precision, recall = evaluate(test_data)
print(
            'test: mic f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


# In[ ]:




