import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np

'''
true_labels = pickle.load(open("true.pkl", 'rb'))
pred_labels = pickle.load(open("pred.pkl", 'rb'))
true_labels_np = np.concatenate(true_labels).ravel()
pred_labels_np = np.concatenate(pred_labels).ravel()

c=0
for a in true_labels_np:
    if a == 1:
        c+=1
print (c)
exit(1)

print ("f1_macro")
print (f1_score(true_labels_np, pred_labels_np, average='macro'))

print ("f1_micro")
print (f1_score(true_labels_np, pred_labels_np, average='micro'))

precision = average_precision_score(true_labels_np, pred_labels_np)
print ("precision")
print (precision)


recall = recall_score(true_labels_np, pred_labels_np)
print ("recall")
print (recall)


accuracy = accuracy_score(true_labels_np, pred_labels_np)
print ("accuracy")
print (accuracy)


exit(1)
'''


import tensorflow as tf
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import os
from loader import load_dataset

from helper import flat_accuracy, format_time

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


device_name = tf.test.gpu_device_name()
device = torch.device("cuda")

batch_size = 1

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

data_dir = "/home1/kchandu/research1/persona_movies/data/categorical_data"

test_inputs, test_masks, test_labels, test_categories = load_dataset(os.path.join(data_dir, "test.all"), tokenizer)
test_data = TensorDataset(test_inputs, test_masks, test_labels, test_categories)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
#model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = BertForSequenceClassification.from_pretrained("./model_save/")
# Tell pytorch to run this model on the GPU.
model.cuda()

model.eval()

# Tracking variables
predictions , true_labels, pred_labels = [], [], []

f = open(os.path.join(data_dir, "test.all"), 'r')
lines = f.readlines()

i, gv = 0, 0
for batch in test_dataloader:
  i+=1
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels, b_categories = batch
  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)
  
  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  predictions.append(logits)
  pred_labels.append( np.argmax(logits, axis=1).flatten() )
  true_labels.append(label_ids)
  pred = np.argmax(logits, axis=1).flatten()
  t = label_ids[0]
  print (lines[gv].strip() + " | "+ str(t) +" | " + str(pred[0]) )
  gv+=1
  #exit(1) 

#pred_labels = np.argmax(predictions, axis=1).flatten()
pred_labels_np = np.concatenate(np.array(pred_labels).flatten().tolist()).ravel()
true_labels_np = np.concatenate(np.array(true_labels).flatten().tolist()).ravel()


print ("f1_macro")
print (f1_score(true_labels_np, pred_labels_np, average='macro'))

print ("f1_micro")
print (f1_score(true_labels_np, pred_labels_np, average='micro'))

precision = average_precision_score(true_labels_np, pred_labels_np)
print ("precision")
print (precision)


recall = recall_score(true_labels_np, pred_labels_np)
print ("recall")
print (recall)


accuracy = accuracy_score(true_labels_np, pred_labels_np)
print ("accuracy")
print (accuracy)













