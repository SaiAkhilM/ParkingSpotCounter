import os
import pandas as pd

# paths
csv_path = "CNRPark_per_image.csv"
img_root = "FULL_IMAGE_1000x750"
output_csv = "cleaned_dataset.csv"

# load csv
df = pd.read_csv(csv_path)

# clean
df = df.dropna(subset=["image_url", "num_cars"])
df = df.drop_duplicates().reset_index(drop=True)

# fix image path
def build_path(image_url):
    # remove "CNR-EXT/PATCHES/"
    relative_path = image_url.replace("CNR-EXT/PATCHES/", "")

    # join with actual folder
    full_path = os.path.join(img_root, relative_path)

    return full_path

df["img_path"] = df["image_url"].apply(build_path)

# check if file exists
def file_exists(path):
    return os.path.exists(path)

df["exists"] = df["img_path"].apply(file_exists)

# keep only valid images
df = df[df["exists"] == True].reset_index(drop=True)

print("after filtering:", len(df))
print("final dataset size:", len(df))

# save cleaned dataset
df[["img_path", "num_cars"]].to_csv(output_csv, index=False)
print("saved cleaned dataset to:", output_csv)
