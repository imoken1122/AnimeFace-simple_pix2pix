import torch as th
import utils
import models
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="")
parser.add_argument("--model_path", default="")
parser.add_argument("--batch_size", default = 32,type=int)
parser.add_argument("--isAug",action="store_true")
opt = parser.parse_args()
model = models.Generator()
model.load_state_dict(th.load(opt.model_path,map_location=th.device('cpu')))

datasets = Dataset(opt.input_path)
dataloader = DataLoader(datasets, batch_size=opt.batch_size,shuffle=True,drop_last=True,num_workers=2)
img,img_sk = next(iter(dataloader))

gen_img=model(img_sk)


utils.plot_generated_image(img,img_sk,gen_img,-1,"train","test")

