from datasets import Dataset
import pickle
from models import Generator,Discriminator,Generator_Original
import torch as th
from torch.utils.data import DataLoader
import torchvision as tv
import glob
from torch import nn,optim
from torchsummary import summary
import statistics
import argparse
import os
from tqdm import tqdm
import utils
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--batch_size", default = 4,type=int)
parser.add_argument("--n_epochs", default = 40,type=int)
parser.add_argument("--continue_train", action="store_true")
parser.add_argument("--noconcat", action="store_false")
parser.add_argument("--every_save_epoch", default = 10,type=int)
parser.add_argument("--cuda",action="store_true")
parser.add_argument("--seed",type=int, default=100)
parser.add_argument("--every_save_image",type=int, default=20)
opt = parser.parse_args()


input_path = opt.input_path
batch_size = opt.batch_size
n_epochs = opt.n_epochs
every_save_epoch = opt.every_save_epoch
continue_train = opt.continue_train
model_name = opt.model_name
every_save_image = opt.every_save_image
cuda = opt.cuda
patch_size=64
start_epochs=0
if not os.path.exists(os.path.expanduser(model_name)):
    os.makedirs(model_name)
utils.setup_logging(["gen","disc"],model_name )

if continue_train: 
    dic = utils.progress_state(model_name=model_name)
    start_epochs = int(dic["epoch"])



device = "cpu"
if cuda:
    device = "cuda"

#th.backend.cudnn.benchmark= True
print("==> Setup Model... ")
G,D = Generator().to(device),Discriminator(3 if opt.noconcat else 4).to(device)
#summary(G,(1,265,256))
#summary(D,(3,265,256))
if continue_train:
    print("loading Genrator and Discriminator model")
    gen_f = sorted(glob.glob(f"{model_name}/models/gen/*") ,reverse=True)[0]
    disc_f = sorted(glob.glob(f"{model_name}/models/disc/*") ,reverse=True)[0]
    print(f"loading Genrator {gen_f} \n Discriminator {disc_f}")
    G.load_state_dict(th.load(gen_f))
    D.load_state_dict(th.load(disc_f))
    
param_G = optim.Adam(G.parameters(),lr = 0.0002, betas=(0.5,0.999))
param_D = optim.Adam(D.parameters(),lr = 0.0002, betas=(0.5,0.999))

ones = th.ones(batch_size,1,patch_size,patch_size).to(device)
zeros = th.zeros(batch_size,1, patch_size,patch_size).to(device)

bce_loss_f = nn.BCEWithLogitsLoss()
LAMBDA = 100.
L1_loss_f = nn.L1Loss()


print("==> Loading Dataset... ")
train_dataloader = DataLoader(Dataset(f"{input_path}/train"), batch_size=batch_size, shuffle=True,drop_last=True,num_workers=2)
test_dataloader = DataLoader(Dataset(f"{input_path}/test"), batch_size=batch_size,shuffle=True, drop_last=True,num_workers=2)

def train():
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []


    for epoch in range(start_epochs, n_epochs+1):
        loss1,loss2,loss3,loss4 = [],[],[],[]
        utils.progress_state(epoch,mode="w",model_name=model_name)
        for i,(img_full, img_sketch) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            img_full,img_sketch =img_full.to(device), img_sketch.to(device)


            #generate rgb image
            gen_img = G(img_sketch)

            # === Learining Discriminator ====
            param_D.zero_grad()
            # expecting img_full predict one 
            if opt.noconcat:
                print(1)
                d_out_1 = D(img_full)
            else:
                print(2)
                d_out_1 = D(th.cat([img_sketch,img_full],dim = 1))
            d_loss_1 = bce_loss_f(d_out_1,ones)

            # expecting gen_img predict zero 
            if opt.noconcat:
                d_out_0 = D(gen_img)
            else:
                d_out_0 = D(th.cat([img_sketch,gen_img],dim = 1))

            d_loss_0 = bce_loss_f(d_out_0,zeros)

            D_loss = (d_loss_0 + d_loss_1)*0.5

            loss4.append(D_loss.item())

            D_loss.backward(retain_graph=True)
            param_D.step()

            # === Learning Generator ====

            param_G.zero_grad()
            if opt.noconcat:
                d_out = D(gen_img)
            else:
                d_out = D(th.cat([img_sketch,gen_img],dim = 1))


            # calc gLoss
            g_loss = bce_loss_f(d_out,ones)
            L1_loss = LAMBDA * L1_loss_f(gen_img,img_full)
            G_loss = g_loss + L1_loss

            loss1.append(g_loss.item())
            loss2.append(L1_loss.item())
            loss3.append(G_loss.item())

            # update Generator param
            G_loss.backward()
            param_G.step()



            # plot
            if i % every_save_image == 0:
                fetch_gen_img = gen_img.detach()
                utils.plot_generated_image(img_full,img_sketch,fetch_gen_img,epoch,"train",model_name)
                val_img,val_img_sk = next(iter(test_dataloader))
                val_img,val_img_sk = val_img.to(device),val_img_sk.to(device)
                val_gen_img = G(val_img_sk).detach()
                utils.plot_generated_image(val_img,val_img_sk,val_gen_img,epoch,"valid",model_name)


        result["log_loss_G_sum"].append(statistics.mean(loss1))
        result["log_loss_G_bce"].append(statistics.mean(loss2))
        result["log_loss_G_mae"].append(statistics.mean(loss3))
        result["log_loss_D"].append(statistics.mean(loss4))
        print(f"Epoch : {epoch} " + f"loss_G_sum = {result['log_loss_G_sum'][-1]} " +
                f"({result['log_loss_G_bce'][-1]}, {result['log_loss_G_mae'][-1]}) " +
                f"log_loss_D = {result['log_loss_D'][-1]}")



        if epoch % every_save_epoch == 0:
            th.save(G.state_dict(),f"{model_name}/models/gen/gen_{epoch}.pth")
            th.save(D.state_dict(),f"{model_name}/models/disc/disc_{epoch}.pth")

    with open(f"{model_name}/loss_log.pkl","wb") as f:
        pickle.dump(result,f)



train()




