from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np

f = open("inference.txt", 'r')
lines = f.readlines()

count = 0
tls, pls = [], []
for line in lines:
    w = line.strip().split("|")
    sent = w[0].strip()
    cat = w[1].strip()
    speaker = w[2].strip()
    tl = int(w[3].strip())
    pl = int(w[4].strip())
    tls.append(tl)
    pls.append(pl)
    if pl == 1 and cat == 'sitc_office':
        count+=1

print (count)



'''
true_labels_np = np.array(tls)
pred_labels_np = np.array(pls)

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
'''








