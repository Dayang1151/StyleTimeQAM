import torch
import numpy as np
# torch.set_printoptions(threshold=np.inf)

def loaddata(data,batch_size):
    sentence, user, label, timestamp, match = data.getdata()
    timestamp = timestamp.tolist()
    for i in range(100):
        timestamp.append(timestamp[-1] + 1)
    out_timestamp = []
    for i in range(len(user)):
        new_Arr = []
        for j in range(i + 1,i + 101):
            new_Arr.append(int(timestamp[j] - timestamp[i]))
        out_timestamp.append(new_Arr)
    out_timestamp = torch.from_numpy(np.array(out_timestamp)).to(torch.long)
    user = torch.from_numpy(user).to(torch.long)
    label = torch.from_numpy(label).to(torch.long)
    match = torch.from_numpy(match).to(torch.long)
    N = len(sentence) // batch_size
    out_sentence = torch.zeros([1,50])
    out_user = torch.zeros([1])
    for i in range(N):
        if (i * batch_size - 100) < 0:
            if i != 0:
                left_0 = sentence[0].repeat(100 - i * batch_size,1)
                left_1 = sentence[:i * batch_size]
                left = torch.cat([left_0,left_1])
                mid = sentence[i * batch_size:(i + 1) * batch_size]
                right = sentence[(i + 1) * batch_size:(i + 1) * batch_size + 100]
            else:
                left = sentence[0].repeat(100,1)
                mid = sentence[i * batch_size:(i + 1) * batch_size]
                right = sentence[(i + 1) * batch_size:(i + 1) * batch_size + 100]
        elif (i + 1) * batch_size + 100 > len(sentence):
            left = sentence[i * batch_size - 100:i * batch_size]
            mid = sentence[i * batch_size:(i + 1) * batch_size]
            right = sentence[(i + 1) * batch_size].repeat(100,1)
        else:
            left = sentence[i * batch_size - 100:i * batch_size]
            mid = sentence[i * batch_size:(i + 1) * batch_size]
            right = sentence[(i + 1) * batch_size:(i + 1) * batch_size + 100]
        conn = torch.cat([left,mid,right])
        out_sentence = torch.cat([out_sentence,conn])

    for i in range(N):
        if ((i + 1) * batch_size + 100) < len(user):
            left = user[i * batch_size:(i + 1) * batch_size]
            right = user[(i + 1) * batch_size:(i + 1) * batch_size + 100]
        else:
            left = user[i * batch_size:(i + 1) * batch_size]
            right1 = user[(i + 1) * batch_size:]
            length = 100 - (user[(i + 1) * batch_size:]).shape[0]
            right2 = torch.tensor([0]).repeat(length)
            right = torch.cat([right1,right2])
        conn = torch.cat([left, right])
        out_user = torch.cat([out_user, conn])


    out_sentence = torch.tensor(out_sentence[1:],dtype=torch.int64)
    out_user = torch.tensor(out_user[1:], dtype=torch.int64)
    return out_sentence,out_user,label,out_timestamp,match