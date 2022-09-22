"""Demo."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 17日 星期四 14:09:56 CST
# ***
# ************************************************************************************/
#

import image_animation

if __name__ == "__main__":
    image_animation.video_predict("videos/0006.mp4", "images/0001.png", "output/image_animation.mp4")
    # image_animation.video_client("PAI", "videos/2.mp4", "images/feynman.jpeg", "output/face_server.mp4")
    # image_animation.video_server("PAI")

