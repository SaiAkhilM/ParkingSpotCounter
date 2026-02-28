import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

csv_path = "CNRPark_per_image.csv"
img_root = "FULL_IMAGE_1000x750"

df = pd.read_csv(csv_path)

df = df.dropna(subset=["image_url", "num_cars"]).drop_duplicates().reset_index(drop=True)

# inspect what image_url looks like
print("example image_url:", df.loc[0, "image_url"])

# test an image
#img = Image.open(df.iloc[0]["img_path"])
#print(img.size)
#img.show()


# NOTES: would we have to build a local path from image_url? TT

# the example image_url from the subset csv gives us : example image_url: CNR-EXT/PATCHES/SUNNY/2015-11-12/camera1/S_2015-11-12_07.09_C01_184.jpg
# but our actual images are in: FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1/...

