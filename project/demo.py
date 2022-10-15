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
    image_animation.face_video_predict("images/0001.png", "videos/0006.mp4", "output/face_motion.mp4")
    # image_animation.body_video_predict(...)
