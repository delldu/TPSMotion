"""Image Animation Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import redos
import todos
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from . import motion

import pdb

FACE_IMAGE_SIZE = 256
BODY_IMAGE_SIZE = 512
MGIF_IMAGE_SIZE = 256


def get_face_model():
    """Create model."""

    model_path = "models/video_drive_face.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = motion.ImageAnimation()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_drive_face.torch"):
        model.save("output/video_drive_face.torch")

    return model, device


def model_forward(model, device, face_tensor, driving_tensor):
    face_tensor = face_tensor.to(device)
    driving_tensor = driving_tensor.to(device)

    with torch.no_grad():
        output_tensor = todos.model.two_forward(model, device, face_tensor, driving_tensor)

    return output_tensor


def face_motion_predict(face_file, video_file, output_file):
    # load video
    video = redos.video.Reader(video_file)
    if video.n_frames < 1:
        print(f"Read video {video_file} error.")
        return False

    # load face image
    face_image = Image.open(face_file).convert("RGB").resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))
    face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_face_model()

    print(f"{video_file} driving {face_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)
    first_driving_tensor = torch.zeros(1, 3, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)

    def clean_video_frame(no, data):
        global first_driving_tensor

        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        driving_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        driving_tensor = driving_tensor[:, 0:3, :, :]
        driving_tensor = todos.data.resize_tensor(driving_tensor, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)

        if no == 0:
            first_driving_tensor = driving_tensor

        driving_tensor = torch.cat((first_driving_tensor, driving_tensor), dim = 1)
        output_tensor = model_forward(model, device, face_tensor, driving_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True
