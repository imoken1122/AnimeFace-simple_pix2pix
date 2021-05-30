import numpy as np
import torch as th
import torch.utils.data as data
from torchvision import transforms
import torchvision as tv
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt
class Dataset(data.Dataset):
    def __init__(self, dir_path ):
        super().__init__()

        self.img_path = glob.glob(f"{dir_path}/*")
        self.tf_full = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
            ])
        self.tf_sketch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)) 
        ])

        self.len = len(self.img_path)

    def make_contour_image(self,gray):
        neiborhood24 = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]],
                                np.uint8)
        dilated = cv2.dilate(gray, neiborhood24, iterations=1)
        diff = cv2.absdiff(dilated, gray)
        contour = 255 - diff

        return contour

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        path = self.img_path[idx]
        img1 = cv2.resize(cv2.imread(path ),(256,256))
        img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

        #img = Image.open(path).resize((256,256))
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #img_sketch = self.make_contour_image(np.asarray(img.convert("L")))
        img_sketch = self.make_contour_image(gray)

        img = self.tf_full(img2)
        img_sketch = self.tf_sketch(img_sketch)

        return img,img_sketch

"""
datasets = Dataset("../../faces/images/train")
datasets = Dataset("../../pixiv_images")
dataloader = data.DataLoader(datasets,batch_size=4,shuffle = True,num_workers = 2,drop_last = True)
for i,j in dataloader:
    pass
img,img_sk=next(iter(dataloader))
print(img.shape,img_sk.shape)
grid1 = tv.utils.make_grid(img)
grid2 = tv.utils.make_grid(img_sk)
grid = th.cat([grid1,grid2],dim=1)
print(grid.shape)

grid = np.transpose(( (grid + 1.) * 127.5)/255,[1,2,0])
print(grid.min(),grid.max(),)
plt.axis("off")
plt.imshow(grid)
plt.savefig("output.png")
#tv.utils.save_image(grid,"output.png")
print(grid.shape)

"""