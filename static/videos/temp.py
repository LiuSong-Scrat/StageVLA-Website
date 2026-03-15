import cv2
import numpy as np
# dir_path = "/home/liusong/桌面/temp/push_buttons/all_variations/episodes/episode24/front_rgb"
# dir_path = "/home/liusong/桌面/temp/put_money_in_safe/all_variations/episodes/episode58/left_shoulder_rgb"
dir_path = "/home/liusong/桌面/temp/close_laptop_lid/mnt/bn/lpy-lq/3D_VLA/Nips2025/colosseum_data/eval/close_laptop_lid/close_laptop_lid_0/variation0/episodes/episode18/front_rgb"
import os 

file_list =  [os.path.join(dir_path,file_name) for file_name in os.listdir(dir_path) if file_name.endswith(".jpg") or file_name.endswith(".png")]
file_int =  [int(file_name[:-4]) for file_name in os.listdir(dir_path) if file_name.endswith(".jpg") or file_name.endswith(".png")]
file_int_arr = np.array(file_int)
file_sort_index = np.argsort(file_int_arr) 

fourcc = cv2.VideoWriter_fourcc(*"mpeg")
frame_rate = 30
resolution = (640, 480)
video_writer = cv2.VideoWriter(os.path.join("/home/liusong/temp/StageVLA-Website","task0.mp4"), fourcc, frame_rate, resolution)


for index in file_sort_index:
    file_path = file_list[index]
    image = cv2.imread(file_path)
    bgr = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
    video_writer.write(bgr)
video_writer.release()
