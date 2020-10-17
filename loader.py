import torch
import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

def load_dataset(file_path, tokenizer):

    sentences, labels, categories = [], [], []
    label_dict = {'arch':0, 'sitc':1}
    cat_dict = {'got':0, 'movies':1, 'once':3, 'glee':4, 'himym':5, 'office':6}
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    except_count = 0
    for line in lines:
        try:
            w = line.strip().split('|')
            text = w[0].strip()
            combo_categories = w[1].strip().split("_")
            label = label_dict[combo_categories[0]]
            category = cat_dict[combo_categories[1]]
            speaker = w[2].strip()
            sentences.append(text)
            labels.append(label)
            categories.append(category)
        except:
            except_count+=1

    print (except_count)
    
    input_ids = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                       )
        input_ids.append(encoded_sent)
    
    print('Max sentence length: ', max([len(sen) for sen in input_ids]))
    MAX_LEN = 45
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels), torch.tensor(categories)


