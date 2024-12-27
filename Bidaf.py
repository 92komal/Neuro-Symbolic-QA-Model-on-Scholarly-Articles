#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch import nn
import torch
import numpy as np
import pandas as pd
import pickle, time
import re, os, string, typing, gc, json
import torch.nn.functional as F
import spacy
from sklearn.model_selection import train_test_split
from collections import Counter
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import English
# from preprocess import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
import csv
import pprint


# **Dataset Load**

# 

# In[ ]:


def make_dataframe(path, labels):
   
    data = []
    context = []
    Answer = []
    question = []
    with open(path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = []
        for row in csv_reader:
            line_count.append(row)
    for i in range(len(line_count)):
      txt = ' '
      for j in range(len(line_count[i])):
        txt = txt + line_count[i][j]
      data.append(txt)
      txt = txt + '\n' 
    ##########################################################
  

    ############################################################
    label_list = []
    label_list_final = []
    final_label = []
    f = open(labels, "r") 
    # print(len(f.readlines()))
    for i in f.read().split('\n'):
      # print(i)
      s = i.replace('\t', ',')
      # print(s)
      label_list.append(s)
      label_list_final = label_list
    # print(label_list_final)

    for i in range(len(label_list_final)):
        k = label_list_final[i].split(",")
        final_label.append(k)

    
    list1 = []
    list2 = []
    list3 = []
    for i in range(len(label_list_final)-1):
      # print(final_label[i])
      answer_start = final_label[i][0]
      # print("s",answer_start)
      answer_end = final_label[i][1]
      # print("e",answer_end)
      list2.append([answer_start, answer_end])
      ####
      # list1.append(final_label[i])
      # answer_start = label_list[i][0]
      # # print("ans", label_list[i])
      # answer_end = list1[i][1]
      # print("ans", label_list[i][1])
      # list2.append([answer_start, answer_end])
      # list3.append(list2)
    # print(list2) 
    # print("list",list2[0:3]) 
    ###########################################################

      # Extract the Context Question Answer
    for i in range(len(line_count)):
      test = data[i].strip().split('\t')
      context.append(test[0])
      question.append(test[2])
      Answer.append(test[1])
 
    
    
    ####################################################################
    #create dictionary
    qa_dict = {}
    qa_list = []
    # qa_dict[''] = id
    qa_dict['context'] = context
    qa_dict['question'] = question
    qa_dict['label'] = list2
    qa_dict['Answer'] = Answer
    qa_list.append(qa_dict)
    #####################################################################
    vstack_array=np.empty([1,4],dtype='object')
    for i in range(len(context)):
      row=[]
      # print("kkkkkk",list2[i])
      row.append(context[i])
      row.append(question[i])
      row.append(list2[i])
      row.append(Answer[i])
      row=np.array(row, dtype='object')
      vstack_array=np.vstack((vstack_array,row))
      l1=vstack_array.tolist()     
    l1=l1[1:]    
   


    # row = []
    # row1=[]
    # row.append(context[0])
    # row.append(question[0])
    # row.append(list2[0])
    # row.append(Answer[0])
    # #print(row)
    # row=np.array(row, dtype='object')
    # print(row)
    # row1.append(context[1])
    # row1.append(question[1])
    # row1.append(list2[1])
    # row1.append(Answer[1])
    # #print(row1)
    # row1=np.array(row1, dtype='object')
    # print(row1)
    # l=np.vstack((row,row1))
    # l1=l.tolist()         
    # print(len(l1))                    
    dframe = pd.DataFrame(l1,columns = ['context', 'question','label','answer'])
    # dframe = pd.DataFrame(np.column_stack([context, question,   Answer]), 
    #                             columns=['context','question',  'Answer'])
    #dframe.head()
    
    return dframe


# In[ ]:


train_list = make_dataframe("/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/Train.csv", "/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/train_span.txt")
test_list = make_dataframe("/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/dev.csv", "/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/dev-span.txt")
valid_list = make_dataframe("/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/test.csv", "/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/test-span.txt")
test_list.head()


# In[ ]:


# train_df = pd.DataFrame(train_list)
# test_df = pd.DataFrame(test_list)
# valid_df = pd.DataFrame(valid_list)
# test_df
import random
train_id = random.sample(range(1, 100000), len(train_list) )
test_id = random.sample(range(1, 100000), len(test_list) )
valid_id = random.sample(range(1, 100000), len(valid_list) )

train_list['id'] = train_id
test_list['id'] = test_id
valid_list['id'] = valid_id

train_list = train_list[0:1]
valid_list = valid_list[0:1]
valid_list.head()


# In[ ]:


train_data=train_list
test_data=test_list
valid_data=valid_list

train_list.head()


# **Data Preprocessing**

# In[ ]:


import sys
def preprocess_df(df):
    
    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df.answer = df.answer.apply(to_lower)


# In[ ]:


preprocess_df(train_data)
preprocess_df(valid_data)
train_data.head()


# In[ ]:


def gather_text_for_vocab(dfs:list):
    '''
    Gathers text from contexts and questions to build a vocabulary.
    
    :param dfs: list of dataframes of SQUAD dataset.
    :returns: list of contexts and questions
    '''
    
    text = []
    total = 0
    for df in dfs:
        unique_contexts = list(df.context.unique())
        unique_questions = list(df.question.unique())
        total += df.context.nunique() + df.question.nunique()
        text.extend(unique_contexts + unique_questions)
    
    assert len(text) == total
    
    return text


# In[ ]:


def build_word_vocab(vocab_text):
    '''
    Builds a word-level vocabulary from the given text.
    
    :param list vocab_text: list of contexts and questions
    :returns 
        dict word2idx: word to index mapping of words
        dict idx2word: integer to word mapping
        list word_vocab: list of words sorted by frequency
    '''
    
    
    words = []
    for sent in vocab_text:
        for word in nlp(sent, disable=['parser','tagger','ner']):
            words.append(word.text)

    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    print(f"raw-vocab: {len(word_vocab)}")
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word:idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v:k for k,v in word2idx.items()}
    
    
    return word2idx, idx2word, word_vocab


# In[ ]:


def build_char_vocab(vocab_text):
    '''
    Builds a character-level vocabulary from the given text.
    
    :param list vocab_text: list of contexts and questions
    :returns 
        dict char2idx: character to index mapping of words
        list char_vocab: list of characters sorted by frequency
    '''
    
    chars = []
    for sent in vocab_text:
        for ch in sent:
            chars.append(ch)

    char_counter = Counter(chars)
    char_vocab = sorted(char_counter, key=char_counter.get, reverse=True)
    print(f"raw-char-vocab: {len(char_vocab)}")
    high_freq_char = [char for char, count in char_counter.items() if count>=20]
    char_vocab = list(set(char_vocab).intersection(set(high_freq_char)))
    print(f"char-vocab-intersect: {len(char_vocab)}")
    char_vocab.insert(0,'<unk>')
    char_vocab.insert(1,'<pad>')
    char2idx = {char:idx for idx, char in enumerate(char_vocab)}
    print(f"char2idx-length: {len(char2idx)}")
    
    return char2idx, char_vocab


# In[ ]:


def context_to_ids(text, word2idx):
    '''
    Converts context text to their respective ids by mapping each word
    using word2idx. Input text is tokenized using spacy tokenizer first.
    
    :param str text: context text to be converted
    :param dict word2idx: word to id mapping

    :returns list context_ids: list of mapped ids
    
    :raises assertion error: sanity check
    
    '''
    
    context_tokens = [w.text for w in nlp(text, disable=['parser','tagger','ner'])]
    context_ids = [word2idx[word] for word in context_tokens]
    
    assert len(context_ids) == len(context_tokens)
    return context_ids


# In[ ]:


def question_to_ids(text, word2idx):
    '''
    Converts question text to their respective ids by mapping each word
    using word2idx. Input text is tokenized using spacy tokenizer first.
    
    :param str text: question text to be converted
    :param dict word2idx: word to id mapping
    :returns list context_ids: list of mapped ids
    
    :raises assertion error: sanity check
    
    '''
    
    question_tokens = [w.text for w in nlp(text, disable=['parser','tagger','ner'])]
    question_ids = [word2idx[word] for word in question_tokens]
    
    assert len(question_ids) == len(question_tokens)
    return question_ids


# In[ ]:


def get_error_indices(df, idx2word):
    # print('Komal', df.head())
    start_value_error, end_value_error, assert_error = test_indices(df, idx2word)
    err_idx = start_value_error + end_value_error + assert_error
    err_idx = set(err_idx)
    print(f"Number of error indices: {len(err_idx)}")
    
    return err_idx


# In[ ]:


def test_indices(df, idx2word):
    '''
    Performs the tests mentioned above. This method also gets the start and end of the answers
    with respect to the context_ids for each example.
    
    :param dataframe df: SQUAD df
    :param dict idx2word: inverse mapping of token ids to words
    :returns
        list start_value_error: example idx where the start idx is not found in the start spans
                                of the text
        list end_value_error: example idx where the end idx is not found in the end spans
                              of the text
        list assert_error: examples that fail assertion errors. A majority are due to the above errors
        
    '''
    
    start_value_error = []
    end_value_error = []
    assert_error = []
    # print(df.head())
    for index, row in df.iterrows():
        # print("index", index)
        # print("row", row)

        answer_tokens = [w.text for  w in nlp(row['answer'], disable=['parser','tagger','ner'])]
        print("komal",answer_tokens)
       
        

        start_token = answer_tokens[0]
        end_token = answer_tokens[-1]
        # print("s", start_token)
        # print("e", end_token)
        
        context_span  = [(word.idx, word.idx + len(word.text)) 
                         for word in nlp(row['context'], disable=['parser','tagger','ner'])]
        print(context_span)
        
        starts, ends = zip(*context_span)
        print("starts", type(starts))
        print("starts11", type(starts[0]))
        print("ends", ends)

        answer_start, answer_end = row['label']
        print("row", row['label'])
        answer_start = int(answer_start)
        answer_end = int(answer_end)
        print("answer_start",type(answer_start))
        print("answer_end", answer_end)

        try:
            start_idx = starts.index(answer_start)
            print("start_idx",start_idx)
        except:
            start_value_error.append(index)
            print("except")
        try:
            end_idx  = ends.index(answer_end)
            print("end_idx", end_idx)
        except:
            end_value_error.append(index)
            print("end_value_error", end_value_error)

        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except:
            assert_error.append(index)


    return start_value_error, end_value_error, assert_error


# In[ ]:


def index_answer(row, idx2word):
    '''
    Takes in a row of the dataframe or one training example and
    returns a tuple of start and end positions of answer by calculating 
    spans.
    '''
    
    context_span = [(word.idx, word.idx + len(word.text)) for word in nlp(row.context, disable=['parser','tagger','ner'])]
    starts, ends = zip(*context_span)
    
    answer_start, answer_end = row.label
    answer_start = int(answer_start)
    answer_end = int(answer_end)
    start_idx = starts.index(answer_start)
 
    end_idx  = ends.index(answer_end)
    
    ans_toks = [w.text for w in nlp(row.answer,disable=['parser','tagger','ner'])]
    ans_start = ans_toks[0]
    ans_end = ans_toks[-1]
    assert idx2word[row.context_ids[start_idx]] == ans_start
    assert idx2word[row.context_ids[end_idx]] == ans_end
    
    return [start_idx, end_idx]


# In[ ]:


train_data.head()


# In[ ]:


# gather text to build vocabularies

get_ipython().run_line_magic('time', 'vocab_text = gather_text_for_vocab([train_data, valid_data])')
print("Number of sentences in dataset: ", len(vocab_text))


# In[ ]:


get_ipython().run_line_magic('time', 'word2idx, idx2word, word_vocab = build_word_vocab(vocab_text)')
print("----------------------------------")
get_ipython().run_line_magic('time', 'char2idx, char_vocab = build_char_vocab(vocab_text)')


# In[ ]:


# numericalize context and questions for training and validation set

get_ipython().run_line_magic('time', "train_data['context_ids'] = train_data.context.apply(context_to_ids, word2idx=word2idx)")
get_ipython().run_line_magic('time', "valid_data['context_ids'] = valid_data.context.apply(context_to_ids, word2idx=word2idx)")
get_ipython().run_line_magic('time', "train_data['question_ids'] = train_data.question.apply(question_to_ids, word2idx=word2idx)")
get_ipython().run_line_magic('time', "valid_data['question_ids'] = valid_data.question.apply(question_to_ids, word2idx=word2idx)")
# print("koml",train_data.head())


# In[ ]:


train_data.head()


# In[ ]:


# get indices with tokenization errors and drop those indices 

train_err = get_error_indices(train_data, idx2word)
valid_err = get_error_indices(valid_data, idx2word)

train_data.drop(train_err, inplace=True)
valid_data.drop(valid_err, inplace=True)


# In[ ]:


valid_data.head()


# In[ ]:


train_data.head()


# In[ ]:


# get start and end positions of answers from the context
# this is basically the label for training QA models

train_label_idx = train_data.apply(index_answer, axis=1, idx2word=idx2word)
valid_label_idx = valid_data.apply(index_answer, axis=1, idx2word=idx2word)
print(train_label_idx)

train_data['label_idx'] = train_label_idx
valid_data['label_idx'] = valid_label_idx


# In[ ]:


# dump to pickle files

train_data.to_pickle('bidaftrain.pkl')
valid_data.to_pickle('bidafvalid.pkl')

with open('qanetw2id.pickle','wb') as handle:
    pickle.dump(word2idx, handle)

with open('qanetc2id.pickle','wb') as handle:
    pickle.dump(char2idx, handle)


# In[ ]:


# load data from pickle files


train_data = pd.read_pickle('bidaftrain.pkl')
valid_data = pd.read_pickle('bidafvalid.pkl')

with open('qanetw2id.pickle','rb') as handle:
    word2idx = pickle.load(handle)
with open('qanetc2id.pickle','rb') as handle:
    char2idx = pickle.load(handle)

idx2word = {v:k for k,v in word2idx.items()}


# In[ ]:


class SquadDataset:
    '''
    - Creates batches dynamically by padding to the length of largest example
      in a given batch.
    - Calulates character vectors for contexts and question.
    - Returns tensors for training.
    '''
    
    def __init__(self, data, batch_size):
        
        self.batch_size = batch_size
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data
        
        
    def __len__(self):
        return len(self.data)
    
    def make_char_vector(self, max_sent_len, max_word_len, sentence):
        
        char_vec = torch.ones(max_sent_len, max_word_len).type(torch.LongTensor)
        
        for i, word in enumerate(nlp(sentence, disable=['parser','tagger','ner'])):
            for j, ch in enumerate(word.text):
                char_vec[i][j] = char2idx.get(ch, 0)
        
        return char_vec    
    
    def get_span(self, text):
        
        text = nlp(text, disable=['parser','tagger','ner'])
        span = [(w.idx, w.idx+len(w.text)) for w in text]

        return span

    def __iter__(self):
        '''
        Creates batches of data and yields them.
        
        Each yield comprises of:
        :padded_context: padded tensor of contexts for each batch 
        :padded_question: padded tensor of questions for each batch 
        :char_ctx & ques_ctx: character-level ids for context and question
        :label: start and end index wrt context_ids
        :context_text,answer_text: used while validation to calculate metrics
        :ids: question_ids for evaluation
        
        '''
        
        for batch in self.data:
            
            spans = []
            ctx_text = []
            answer_text = []
            
            for ctx in batch.context:
                ctx_text.append(ctx)
                spans.append(self.get_span(ctx))
            
            for ans in batch.answer:
                answer_text.append(ans)
                
            
            max_context_len = max([len(ctx) for ctx in batch.context_ids])
            padded_context = torch.LongTensor(len(batch), max_context_len).fill_(1)
            
            for i, ctx in enumerate(batch.context_ids):
                padded_context[i, :len(ctx)] = torch.LongTensor(ctx)
                
            max_word_ctx = 0
            for context in batch.context:
                for word in nlp(context, disable=['parser','tagger','ner']):
                    if len(word.text) > max_word_ctx:
                        max_word_ctx = len(word.text)
            
            char_ctx = torch.ones(len(batch), max_context_len, max_word_ctx).type(torch.LongTensor)
            for i, context in enumerate(batch.context):
                char_ctx[i] = self.make_char_vector(max_context_len, max_word_ctx, context)
            
            max_question_len = max([len(ques) for ques in batch.question_ids])
            padded_question = torch.LongTensor(len(batch), max_question_len).fill_(1)
            
            for i, ques in enumerate(batch.question_ids):
                padded_question[i, :len(ques)] = torch.LongTensor(ques)
                
            max_word_ques = 0
            for question in batch.question:
                for word in nlp(question, disable=['parser','tagger','ner']):
                    if len(word.text) > max_word_ques:
                        max_word_ques = len(word.text)
            
            char_ques = torch.ones(len(batch), max_question_len, max_word_ques).type(torch.LongTensor)
            for i, question in enumerate(batch.question):
                char_ques[i] = self.make_char_vector(max_question_len, max_word_ques, question)
            
            ids = list(batch.id)  
            label = torch.LongTensor(list(batch.label_idx))
            
            yield (padded_context, padded_question, char_ctx, char_ques, label, ctx_text, answer_text, ids)
            
            


# In[ ]:


def p(self, b=None):
    if b is None:
        b = self.a
    print(b)


# In[ ]:


train_dataset = SquadDataset(train_data, 16)


# In[ ]:


valid_dataset = SquadDataset(valid_data, 16)


# In[ ]:


a = next(iter(train_dataset))


# In[ ]:


weights_matrix = np.load('/Data/komal/komal_bidaf/Logic/bidafglove_tv.npy')
num_embeddings, embedding_dim = weights_matrix.shape
#embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=True)
embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix),freeze=True)


# In[ ]:


def get_glove_dict():
    '''
    Parses the glove word vectors text file and returns a dictionary with the words as
    keys and their respective pretrained word vectors as values.

    '''
    glove_dict = {}
    with open("/Data/komal/komal_bidaf/Logic/glove.6B.100d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_dict[word] = vector
            
    f.close()
    
    return glove_dict


# In[ ]:


glove_dict = get_glove_dict()


# In[ ]:





# In[ ]:


def create_weights_matrix(glove_dict):
    '''
    Creates a weight matrix of the words that are common in the GloVe vocab and
    the dataset's vocab. Initializes OOV words with a zero vector.
    '''
    weights_matrix = np.zeros((len(word_vocab), 100))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except:
            pass
        
    return weights_matrix, words_found


# In[ ]:


weights_matrix, words_found = create_weights_matrix(glove_dict)
print("Words found in the GloVe vocab: " ,words_found)


# In[ ]:


# dump the weights to load in future

np.save('/Data/komal/komal_bidaf/Logic/bidafglove_tv.npy', weights_matrix)


# In[ ]:


class CharacterEmbeddingLayer(nn.Module):
    
    def __init__(self, char_vocab_dim, char_emb_dim, num_output_channels, kernel_size):
        
        super().__init__()
        
        self.char_emb_dim = char_emb_dim
        
        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim, padding_idx=1)
        
        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=kernel_size)
        
        self.relu = nn.ReLU()
    
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x = [bs, seq_len, word_len]
        # returns : [batch_size, seq_len, num_output_channels]
        # the output can be thought of as another feature embedding of dim 100.
        
        batch_size = x.shape[0]
        
        x = self.dropout(self.char_embedding(x))
        # x = [bs, seq_len, word_len, char_emb_dim]
        
        # following three operations manipulate x in such a way that
        # it closely resembles an image. this format is important before 
        # we perform convolution on the character embeddings.
        
        x = x.permute(0,1,3,2)
        # x = [bs, seq_len, char_emb_dim, word_len]
        
        x = x.view(-1, self.char_emb_dim, x.shape[3])
        # x = [bs*seq_len, char_emb_dim, word_len]
        
        x = x.unsqueeze(1)
        # x = [bs*seq_len, 1, char_emb_dim, word_len]
        
        # x is now in a format that can be accepted by a conv layer. 
        # think of the tensor above in terms of an image of dimension
        # (N, C_in, H_in, W_in).
        
        x = self.relu(self.char_convolution(x))
        # x = [bs*seq_len, out_channels, H_out, W_out]
        
        x = x.squeeze()
        # x = [bs*seq_len, out_channels, W_out]
                
        x = F.max_pool1d(x, x.shape[2]).squeeze()
        # x = [bs*seq_len, out_channels, 1] => [bs*seq_len, out_channels]
        
        x = x.view(batch_size, -1, x.shape[-1])
        # x = [bs, seq_len, out_channels]
        # x = [bs, seq_len, features] = [bs, seq_len, 100]
        
        
        return x        


# In[ ]:


class HighwayNetwork(nn.Module):
    
    def __init__(self, input_dim, num_layers=2):
        
        super().__init__()
        
        self.num_layers = num_layers
        
        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        
        for i in range(self.num_layers):
            
            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))
            
            x = gate_value * flow_value + (1-gate_value) * x
        
        return x


# In[ ]:


class ContextualEmbeddingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.highway_net = HighwayNetwork(input_dim)
        
    def forward(self, x):
        # x = [bs, seq_len, input_dim] = [bs, seq_len, emb_dim*2]
        # the input is the concatenation of word and characeter embeddings
        # for the sequence.
        
        highway_out = self.highway_net(x)
        # highway_out = [bs, seq_len, input_dim]
        
        outputs, _ = self.lstm(highway_out)
        # outputs = [bs, seq_len, emb_dim*2]
        
        return outputs


# In[ ]:


class BiDAF(nn.Module):
    
    def __init__(self, char_vocab_dim, emb_dim, char_emb_dim, num_output_channels, 
                 kernel_size, ctx_hidden_dim, device):
        '''
        char_vocab_dim = len(char2idx)
        emb_dim = 100
        char_emb_dim = 8
        num_output_chanels = 100
        kernel_size = (8,5)
        ctx_hidden_dim = 100
        '''
        super().__init__()
        
        self.device = device
        
        self.word_embedding = self.get_glove_embedding()
        
        self.character_embedding = CharacterEmbeddingLayer(char_vocab_dim, char_emb_dim, 
                                                      num_output_channels, kernel_size)
        
        self.contextual_embedding = ContextualEmbeddingLayer(emb_dim*2, ctx_hidden_dim)
        
        self.dropout = nn.Dropout()
        
        self.similarity_weight = nn.Linear(emb_dim*6, 1, bias=False)
        
        self.modeling_lstm = nn.LSTM(emb_dim*8, emb_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        
        self.output_start = nn.Linear(emb_dim*10, 1, bias=False)
        
        self.output_end = nn.Linear(emb_dim*10, 1, bias=False)
        
        self.end_lstm = nn.LSTM(emb_dim*2, emb_dim, bidirectional=True, batch_first=True)
        
    
    def get_glove_embedding(self):
        
        weights_matrix = np.load('/Data/komal/komal_bidaf/Logic/bidafglove_tv.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=True)

        return embedding
        
    def forward(self, ctx, ques, char_ctx, char_ques):
        # ctx = [bs, ctx_len]
        # ques = [bs, ques_len]
        # char_ctx = [bs, ctx_len, ctx_word_len]
        # char_ques = [bs, ques_len, ques_word_len]
        
        ctx_len = ctx.shape[1]
        
        ques_len = ques.shape[1]
        
        ## GET WORD AND CHARACTER EMBEDDINGS
        
        ctx_word_embed = self.word_embedding(ctx)
        # ctx_word_embed = [bs, ctx_len, emb_dim]
        
        ques_word_embed = self.word_embedding(ques)
        # ques_word_embed = [bs, ques_len, emb_dim]
        
        ctx_char_embed = self.character_embedding(char_ctx)
        # ctx_char_embed =  [bs, ctx_len, emb_dim]
        
        ques_char_embed = self.character_embedding(char_ques)
        # ques_char_embed = [bs, ques_len, emb_dim]
        
        ## CREATE CONTEXTUAL EMBEDDING
        
        ctx_contextual_inp = torch.cat([ctx_word_embed, ctx_char_embed],dim=2)
        # [bs, ctx_len, emb_dim*2]
        
        ques_contextual_inp = torch.cat([ques_word_embed, ques_char_embed],dim=2)
        # [bs, ques_len, emb_dim*2]
        
        ctx_contextual_emb = self.contextual_embedding(ctx_contextual_inp)
        # [bs, ctx_len, emb_dim*2]
        
        ques_contextual_emb = self.contextual_embedding(ques_contextual_inp)
        # [bs, ques_len, emb_dim*2]
        
        
        ## CREATE SIMILARITY MATRIX
        
        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1,1,ques_len,1)
        # [bs, ctx_len, 1, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        
        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1,ctx_len,1,1)
        # [bs, 1, ques_len, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        
        elementwise_prod = torch.mul(ctx_, ques_)
        # [bs, ctx_len, ques_len, emb_dim*2]
        
        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        # [bs, ctx_len, ques_len, emb_dim*6]
        
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        # [bs, ctx_len, ques_len]
        
        
        ## CALCULATE CONTEXT2QUERY ATTENTION
        
        a = F.softmax(similarity_matrix, dim=-1)
        # [bs, ctx_len, ques_len]
        
        c2q = torch.bmm(a, ques_contextual_emb)
        # [bs] ([ctx_len, ques_len] X [ques_len, emb_dim*2]) => [bs, ctx_len, emb_dim*2]
        
        
        ## CALCULATE QUERY2CONTEXT ATTENTION
        
        b = F.softmax(torch.max(similarity_matrix,2)[0], dim=-1)
        # [bs, ctx_len]
        
        b = b.unsqueeze(1)
        # [bs, 1, ctx_len]
        
        q2c = torch.bmm(b, ctx_contextual_emb)
        # [bs] ([bs, 1, ctx_len] X [bs, ctx_len, emb_dim*2]) => [bs, 1, emb_dim*2]
        
        q2c = q2c.repeat(1, ctx_len, 1)
        # [bs, ctx_len, emb_dim*2]
        
        ## QUERY AWARE REPRESENTATION
        
        G = torch.cat([ctx_contextual_emb, c2q, 
                       torch.mul(ctx_contextual_emb,c2q), 
                       torch.mul(ctx_contextual_emb, q2c)], dim=2)
        
        # [bs, ctx_len, emb_dim*8]
        
        
        ## MODELING LAYER
        
        M, _ = self.modeling_lstm(G)
        # [bs, ctx_len, emb_dim*2]
        
        ## OUTPUT LAYER
        
        M2, _ = self.end_lstm(M)
        
        # START PREDICTION
        
        p1 = self.output_start(torch.cat([G,M], dim=2))
        # [bs, ctx_len, 1]
        
        p1 = p1.squeeze()
        # [bs, ctx_len]
        
        #p1 = F.softmax(p1, dim=-1)
        
        # END PREDICTION
        
        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        # [bs, ctx_len, 1] => [bs, ctx_len]
        
        #p2 = F.softmax(p2, dim=-1)
        
        
        return p1, p2
    


# In[ ]:


CHAR_VOCAB_DIM = len(char2idx)
EMB_DIM = 100
CHAR_EMB_DIM = 8
NUM_OUTPUT_CHANNELS = 100
KERNEL_SIZE = (8,5)
HIDDEN_DIM = 100
device = torch.device('cuda')

model = BiDAF(CHAR_VOCAB_DIM, 
              EMB_DIM, 
              CHAR_EMB_DIM, 
              NUM_OUTPUT_CHANNELS, 
              KERNEL_SIZE, 
              HIDDEN_DIM, 
              device)


# In[ ]:


import torch.optim as optim
from torch.autograd import Variable
optimizer = optim.Adadelta(model.parameters())


# In[ ]:


def train(model, train_dataset):
    print("Starting training ........")
   

    train_loss = 0.
    batch_count = 0
    model.train()
    for batch in train_dataset:
        
        optimizer.zero_grad()
    
        if batch_count % 500 == 0:
            print(f"Starting batch: {batch_count}")
        batch_count += 1
        
        context, question, char_ctx, char_ques, label, ctx_text, ans, ids = batch

        context, question, char_ctx, char_ques, label = context.to(device), question.to(device),                                   char_ctx.to(device), char_ques.to(device), label.to(device)


        preds = model(context, question, char_ctx, char_ques)

        start_pred, end_pred = preds

        s_idx, e_idx = label[:,0], label[:,1]

        loss = F.cross_entropy(start_pred, s_idx) + F.cross_entropy(end_pred, e_idx)

        loss.backward()
        
        plot_grad_flow(model.named_parameters())
        
        for name, param in model.named_parameters():
            if(param.requires_grad) and ("bias" not in name):
                writer.add_histogram(name+'_grad',param.grad.abs().mean())
    

        optimizer.step()

        train_loss += loss.item()

    return train_loss/len(train_dataset)


# In[ ]:


def valid(model, valid_dataset):
    
    print("Starting validation .........")
   
    valid_loss = 0.

    batch_count = 0
    
    f1, em = 0., 0.
    
    model.eval()
        
   
    predictions = {}
    
    for batch in valid_dataset:

        if batch_count % 500 == 0:
            print(f"Starting batch {batch_count}")
        batch_count += 1

        context, question, char_ctx, char_ques, label, ctx, answers, ids = batch

        context, question, char_ctx, char_ques, label = context.to(device), question.to(device),                                   char_ctx.to(device), char_ques.to(device), label.to(device)
        
       

        
        with torch.no_grad():
            
            s_idx, e_idx = label[:,0], label[:,1]

            preds = model(context, question, char_ctx, char_ques)

            p1, p2 = preds

            
            loss = F.cross_entropy(p1, s_idx) + F.cross_entropy(p2, e_idx)

            valid_loss += loss.item()

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
            
           
            for i in range(batch_size):
                id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i]+1]
                pred = ' '.join([idx2word[idx.item()] for idx in pred])
                predictions[id] = pred
            

    
    em, f1 = evaluate(predictions)
    return valid_loss/len(valid_dataset), em, f1


# In[ ]:


def evaluate(predictions):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple 
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1). 
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the 
    predictions to calculate em, f1.
    
    
    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth 
      match exactly, 0 otherwise.
    : f1_score: 
    '''
    with open('./data/squad_dev.json','r',encoding='utf-8') as f:
        dataset = json.load(f)
        
    dataset = dataset['data']
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue
                
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                
                prediction = predictions[qa['id']]
                
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return exact_match, f1


# In[ ]:


def normalize_answer(s):
    '''
    Performs a series of cleaning steps on the ground truth and 
    predicted answer.
    '''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.
    
    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of 
                               metrics are chosen.
                            
    
    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
        
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:



train_losses = []
valid_losses = []
ems = []
f1s = []
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    start_time = time.time()
    
    train_loss = train(model, train_dataset)
    valid_loss, em, f1 = valid(model, valid_dataset)
    
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('valid_loss', valid_loss, epoch)
    
    
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'em':em,
            'f1':f1,
            }, 'bidaf_run4_{}.pth'.format(epoch))
    
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    ems.append(em)
    f1s.append(f1)

    print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
    print(f"Epoch valid loss: {valid_loss}")
    print(f"Epoch EM: {em}")
    print(f"Epoch F1: {f1}")
    print("====================================================================================")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **TO DO**

# In[ ]:


label_list = []
label_list_final = []
final_label = []
f = open("/Data/komal/komal_bidaf/Logic/ScholarlyRead/ScholarlyRead/dev-span.txt", "r") 
# print(len(f.readlines()))
for i in f.read().split('\n'):
 
  s = i.replace('\t', ',')
  label_list.append(s)
  label_list_final = label_list


for i in range(len(label_list_final)):
  k = label_list_final[i].split(",")
  final_label.append(k)
#
answer_start = []
print(final_label)
# answer_start, answer_end = final_label[0]
for i in range(len(label_list_final)-1):
  print(final_label[i])
  answer_start = final_label[i][0]
  print("s",answer_start)
  answer_end = final_label[i][1]
  print("e",answer_end)




# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:





# In[ ]:





# ############################################################
# 

# In[ ]:


# ################################################################
# #Read label
# label_list = []
# label_list_final = []
# f = open("/content/drive/MyDrive/Logic/ScholarlyRead/ScholarlyRead/Train-span.txt", "r")
# # print(len(f.readlines()))
# for i in f.read().split('\n'):
#   # print(i)
#   label_list.append(i)
# label_list = label_list[:-3]
# print(label_list)
# # for i in label_list:
# #   i.replace('\t', ',')
# #   label_list_final.append(i)


# In[ ]:


# import csv
# import pandas as pd
# import numpy as np
   


# ##################################################################
# #Read CSV file
# def get_data_file():
#     with open("/content/drive/MyDrive/Logic/ScholarlyRead/ScholarlyRead/Train.csv", encoding="utf8") as csv_file:
#         csv_reader = csv.reader(csv_file)
#         line_count = []
#         for row in csv_reader:
#             line_count.append(row)
# ####################################################################        
# context = []
# question = []
# Answer = []
# data = []
# label = []
# for i in range(len(line_count)):
# 	txt = ' '
# 	for j in range(len(line_count[i])):
# 		txt = txt + line_count[i][j]
# 	data.append(txt)
# 	txt = txt + '\n'


# ###################################################################	
# # Extract the Context Question Answer
# for i in range(len(line_count)):
	
# 	test = data[i].strip().split('\t')
	
# 	context.append(test[0])
# 	question.append(test[2])
# 	Answer.append(test[1])
# print(type(context))
# ####################################################################
# #Append list in to the DataFrame	
		
# dframe = pd.DataFrame(np.column_stack([context, question, label_list, Answer]), 
#                                columns=['context', 'question', 'label' , 'Answer'])

# # print(dframe.Answer.to_string(index=False))
# # print(dframe.question.to_string(index=True))

# ##################################################################
# #delete RoW
# dframe = dframe.drop([0], axis=0)
# # print(dframe.head(5))
# # print(dframe['label'])
# #################################################################
# dframe.head()

