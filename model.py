import pandas as pd
from PIL import Image
import os

# paths 
# subset the data
csv_path = "CNRPark_per_image.csv"
img_root = "FULL_IMAGE_1000x750"

# load + clean csv
df = pd.read_csv(csv_path)

#keep only rows we can use
df = df.dropna(subset=["camera", "weather", "year", "month", "day", "hour", "minute", "num_cars"])
df = df.drop_duplicates()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Rows after cleaning:", len(df))
print("Example row:\n", df.iloc[0])