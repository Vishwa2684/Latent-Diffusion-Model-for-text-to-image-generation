from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import json


class CocoWithAnnotations(Dataset):
    def __init__(self,path,tokenizer,transform,train=True):
        super().__init__()
        self.path = path
        self.data = None
        self.transform = transform
        self.tokenizer = tokenizer
        self.train = train
        if self.data is None:
            self.open_json()
    
    def open_json(self):
        if self.train:
            print('======================= Loading training annotations =======================')
            with open(f'{self.path}/annotations/captions_train2014.json','r+') as stream:
                self.data = json.load(stream)
            self.data = self.data['annotations']    
        else:
            print('======================= Loading validation annotations =======================')
            with open(f'./{self.path}/annotations/captions_val2014.json','r+') as stream:
                self.data = json.load(stream)
            self.data = self.data['annotations']
        print('======================= ANNOTATIONS LOADED =======================')
        
    def __getitem__(self, index):
        
        annot = self.data[index]
        if(len(str(annot['image_id']))<6):
            rem_0l = 6-len(str(annot['image_id']))
            rem_0 = ''
            for i in range(rem_0l):
                rem_0+='0'
            image = self.transform(Image.open(f'{self.path}/train2014/COCO_train2014_000000{rem_0+str(annot['image_id'])}.jpg'))
            text_emb = self.tokenizer(annot['caption'])
            return image,text_emb
        else:
            image = self.transform(Image.open(f'{self.path}/train2014/COCO_train2014_000000{annot['image_id']}.jpg'))
            text_emb = self.tokenizer(annot['caption'])
            return image,text_emb