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
BODY_IMAGE_SIZE = 384
MGIF_IMAGE_SIZE = 256

def get_face_model():
    """Create model."""

    model = motion.ImageAnimation(model_path="models/drive_face.pth")
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/video_drive_face.torch"):
        model.save("output/video_drive_face.torch")

    return model, device


def drive_face(video_file, face_file, output_file):
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

    def drive_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        driving_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        driving_tensor = driving_tensor[:, 0:3, :, :]
        driving_tensor = todos.data.resize_tensor(driving_tensor, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)

        output_tensor = todos.model.two_forward(model, device, driving_tensor, face_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=drive_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True


def get_body_model():
    """Create model."""

    model = motion.ImageAnimation(model_path="models/drive_body.pth")
    device = todos.model.get_device()
    model = model.to(device)
    # model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/video_drive_body.torch"):
        model.save("output/video_drive_body.torch")

    return model, device


def drive_body(video_file, body_file, output_file):
    # load video
    video = redos.video.Reader(video_file)
    if video.n_frames < 1:
        print(f"Read video {video_file} error.")
        return False

    # load face image
    body_image = Image.open(body_file).convert("RGB").resize((BODY_IMAGE_SIZE, BODY_IMAGE_SIZE))
    body_tensor = transforms.ToTensor()(body_image).unsqueeze(0)

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_body_model()

    print(f"{video_file} driving {body_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def drive_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        driving_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        driving_tensor = driving_tensor[:, 0:3, :, :]
        driving_tensor = todos.data.resize_tensor(driving_tensor, BODY_IMAGE_SIZE, BODY_IMAGE_SIZE)

        output_tensor = todos.model.two_forward(model, device, driving_tensor, body_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=drive_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True

def get_mgif_model():
    """Create model."""

    model = motion.ImageAnimation(model_path="models/drive_mgif.pth")
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/video_drive_mgif.torch"):
        model.save("output/video_drive_face.torch")

    return model, device