from os import removedirs
import torch
import numpy as np
from sklearn import preprocessing

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin

def decode_prediction(preds, encoder, keep_raw=False):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()

    cap_preds = []
    for j in range(preds.shape[0]):
        temp=[]
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("^")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        
        tp = "".join(temp)

        if not keep_raw:
            tp = remove_duplicates(tp).replace("^","")
        cap_preds.append(tp)
    return cap_preds

def get_lbl_encoder():
    all_chars =  list(np.arange(ord("a") , ord("z") + 1))
    all_chars.extend(list(np.arange(ord("A") , ord("Z") + 1)))
    all_chars.extend(list(np.arange(ord("0") , ord("9") + 1)))
    all_chars = [chr(c) for c in all_chars]

    lbl_encoder = preprocessing.LabelEncoder()
    lbl_encoder.fit(all_chars)
    return lbl_encoder

if __name__=="__main__":
   s = remove_duplicates("^^^^^^^^^^4^^^^y^^^^^c^^^^8^^^^5^^^^^^^^^^^^^^^^^^")
   print(s)
   