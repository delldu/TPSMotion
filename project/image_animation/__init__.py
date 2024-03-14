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

def get_drive_face_keypoint_model():
    """Create model."""

    model = motion.KeyPointDetector(model_path="models/drive_face.pth")
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
    if not os.path.exists("output/drive_face_keypoint.torch"):
        model.save("output/drive_face_keypoint.torch")

    return model, device


def get_drive_face_generator_trace_model():
    """Create model."""

    model = motion.ImageAnimation(model_path="models/drive_face.pth")
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    return model, device


def get_drive_face_generator_model():
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
    if not os.path.exists("output/drive_face_generator.torch"):
        model.save("output/drive_face_generator.torch")

    return model, device


def drive_face(video_file, face_file, output_file):
    # load video
    video = redos.video.Reader(video_file)
    if video.n_frames < 1:
        print(f"Read video {video_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_drive_face_generator_model()
    kpdet, kpdev = get_drive_face_keypoint_model()
    kpdet = kpdet.to(device)
    print(f"Running on {device} ...")

    # load face image
    face_image = Image.open(face_file).convert("RGB").resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))
    face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)
    face_tensor = face_tensor.to(device)
    face_kp = todos.model.forward(kpdet, device, face_tensor)
    face_kp = face_kp.to(device)
    start_driving_kp = torch.zeros_like(face_kp)

    print(f"{video_file} driving {face_file}, save to {output_file} ...")

    progress_bar = tqdm(total=video.n_frames)

    def drive_video_frame(no, data):
        # print(f"-------> frame: {no} -- {data.shape}")
        progress_bar.update(1)

        driving_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        driving_tensor = driving_tensor[:, 0:3, :, :]
        driving_tensor = todos.data.resize_tensor(driving_tensor, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)
        driving_kp = todos.model.forward(kpdet, device, driving_tensor)
        driving_kp = driving_kp.to(device)
        if no == 0:
            global start_driving_kp
            start_driving_kp = driving_kp # save for next step offset kp

        offset_driving_kp = driving_kp - start_driving_kp
        # offset_driving_kp = offset_driving_kp.tanh()/2.0

        with torch.no_grad():
            output_tensor = model(face_kp, offset_driving_kp, face_tensor)

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

def get_drive_body_keypoint_model():
    """Create model."""

    model = motion.KeyPointDetector(model_path="models/drive_body.pth")
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
    if not os.path.exists("output/drive_body_keypoint.torch"):
        model.save("output/drive_body_keypoint.torch")

    return model, device

def get_drive_body_generator_model():
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
    if not os.path.exists("output/drive_body_generator.torch"):
        model.save("output/drive_body_generator.torch")

    return model, device


def drive_body(video_file, body_file, output_file):
    # load video
    video = redos.video.Reader(video_file)
    if video.n_frames < 1:
        print(f"Read video {video_file} error.")
        return False

    model, device = get_drive_body_generator_model()
    kpdet, kpdev = get_drive_body_keypoint_model()
    kpdet = kpdet.to(device)
    print(f"Running on {device} ...")

    # load face image
    body_image = Image.open(body_file).convert("RGB").resize((BODY_IMAGE_SIZE, BODY_IMAGE_SIZE))
    body_tensor = transforms.ToTensor()(body_image).unsqueeze(0)
    body_tensor = body_tensor.to(device)
    body_kp = todos.model.forward(kpdet, device, body_tensor)
    body_kp = body_kp.to(device)
    start_driving_kp = torch.zeros_like(body_kp)

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)


    print(f"{video_file} driving {body_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def drive_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        driving_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        driving_tensor = driving_tensor[:, 0:3, :, :]
        driving_tensor = todos.data.resize_tensor(driving_tensor, BODY_IMAGE_SIZE, BODY_IMAGE_SIZE)

        driving_kp = todos.model.forward(kpdet, device, driving_tensor)
        driving_kp = driving_kp.to(device)
        if no == 0:
            global start_driving_kp
            start_driving_kp = driving_kp # save for next step offset kp

        offset_driving_kp = driving_kp - start_driving_kp
        offset_driving_kp = offset_driving_kp.tanh()/2.0

        with torch.no_grad():
            output_tensor = model(body_kp, offset_driving_kp, body_tensor)

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