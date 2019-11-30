import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model import MultiClassifier
import io
import argparse

def get_model(path):
    model = MultiClassifier()
    model = torch.load(path, map_location='cpu')
    model = model.eval()
    return model



def get_tensor(img):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    return tfms(Image.open((img))).unsqueeze(0)


def predict(img):
    model = get_model('models/yourmodelname')

    label_lst = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair',
    'Blond_Hair', 'Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup',
    'High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
    'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat',
    'Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']


    tnsr = get_tensor(img)
    op = model(tnsr)
    op_b = torch.round(op)
    
    op_b_np = torch.Tensor.cpu(op_b).detach().numpy()
    
    preds = np.where(op_b_np == 1)[1]

    sigs_op = torch.Tensor.cpu(torch.round((op)*100)).detach().numpy()[0]

    o_p = np.argsort(torch.Tensor.cpu(op).detach().numpy())[0][::-1]

    
    
    # print(preds)
    
    label = []
    for i in preds:
        label.append(label_lst[i])

    arg_s = {}
    for i in o_p:
        arg_s[label_lst[int(i)]] = sigs_op[int(i)]

    # return str(label) + "@" + str([(list(arg_s.items())[:10])])
    _l = list(arg_s.items())[:10]

    cd = [': '.join(map(str, tup)) for tup in _l]
    cd = '-'.join(cd)
    # print("CD: {}".format(cd))

    # ', '.join(map(str, tups))

    return str(label)+"@", cd

# lb, oo = predict('/Users/vatsalsaglani/Desktop/njunk/personal/CelebA_API/vst.jpg')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='predict arguments')
    parser.add_argument('img_path', type = str, help = 'Image Required')
    args = parser.parse_args()
    img_path = args.img_path

    l = predict(img_path)

    for i in l:
        print(str(i))