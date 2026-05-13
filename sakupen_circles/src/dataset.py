import torch

import os
from PIL import Image
from tqdm.auto import tqdm

from torch.utils.data import Dataset

class JetbotDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        self.images = dict()

        for idx in tqdm(range(len(self.df))):

            row = self.df.iloc[idx]

            image_name = row["frame_timestamp"] + ".png"

            image_path = os.path.join(
                self.image_dir,
                image_name
            )

            image = Image.open(image_path).convert("RGB")

            # store PIL image in RAM
            self.images[image_name] = image.copy()

            image.close()
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # adjust column name if needed
        image_name = row["frame_timestamp"] + ".png"
        image = self.images[image_name]

        if self.transform:
            image = self.transform(image)

        # targets
        throttle = torch.tensor(row["throttle"], dtype=torch.float32)
        turn = torch.tensor(row["turn"], dtype=torch.float32)

        target = torch.tensor(
            [throttle, turn],
            dtype=torch.float32
        )

        return image, target