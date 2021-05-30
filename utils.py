import os
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision as tv
import json
def create_dir(dirlist):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """
    for dirs in dirlist:
        if isinstance(dirs, (list, tuple)):
            for d in dirs:
                if not os.path.exists(os.path.expanduser(d)):
                    os.makedirs(d)
        elif isinstance(dirs, str):
            if not os.path.exists(os.path.expanduser(dirs)):
                os.makedirs(dirs)


def setup_logging(model_list, model_name='./log'):
    
    # Output path where we store experiment log and weights
    model_dir = [os.path.join(model_name, 'models', mn) for mn in model_list]

    fig_dir = os.path.join(model_name, 'figures')
    
    # Create if it does not exist
    create_dir([model_dir, fig_dir])

def progress_state(epoch=None,mode="r",model_name="./log"):
    dic = {"epoch":""}
    if mode == "w":
        with open(f"{model_name}/setup.json",mode) as f:
            dic["epoch"] = str(epoch)
            json.dump(dic,f)
    else:
        f = open(f"{model_name}/setup.json",mode) 
        dic = json.load(f)
        return dic



def plot_generated_image(img_full,img_sketch,img_gen,epoch,suffix, model_name):
    grid1 = tv.utils.make_grid(img_full[:])
    grid2 = tv.utils.make_grid(img_sketch[:])
    grid3 = tv.utils.make_grid(img_gen[:])
    grid = th.cat([grid1,grid2,grid3],dim=1)
    grid = grid.detach().cpu().numpy()
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"{model_name}/figures/{suffix}_{epoch}.png")
