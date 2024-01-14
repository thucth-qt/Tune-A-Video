import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange

import cv2 
import torch
from glob import glob

class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            *args,
            **kargs
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example

class ImagesDataset(Dataset):
    def __init__(
            self,
            images_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            *args,
            **kargs
    ):
        self.images_path = images_path
        self.prompt = prompt
        self.prompt_ids = None
        self.images = None
        self.width = width
        self.height = height

    def __len__(self):
        return 1

    def __transform(self, image):
        #center cropping into square
        h, w, _ = image.shape
        if h > w:
            image = image[:, (h - w) // 2:(h - w) // 2 + w, :]
        else:
            image = image[(h - w) // 2:(h - w) // 2 + w, :, :]
        
        
        #resize into expected size 
        image = cv2.resize(image, (self.width, self.height))
        return image
        
    def __getitem__(self, index):
        if self.images is None:
            images_path = glob(self.images_path+"/*")
            images = [cv2.imread(path) for path in images_path]
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            images = [self.__transform(image) for image in images]
            self.images = torch.tensor(images, dtype=torch.float32)
            self.images = torch.permute(self.images, (0, 3, 1, 2))
        example = {
            "pixel_values": (self.images / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example
