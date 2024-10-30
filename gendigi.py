import cv2
import os
import json
import shutil
import glob
from tqdm import tqdm
import pickle
import torch
from musetalk.utils.blending import get_image_prepare_material
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.models.vae import VAE

vae = VAE(model_path = "./models/sd-vae-ft-mse/")
vae.vae = vae.vae.half()


with open('prepare.json', 'r') as f:
    prepare = json.load(f)

avatar_id = prepare["avatar_id"]
video_path = prepare["video_path"]
bbox_shift = prepare["bbox_shift"]
avatar_path = f"Avatars/{avatar_id}"
full_imgs_path = f"{avatar_path}/full_imgs" 
coords_path = f"{avatar_path}/coords.pkl"
latents_out_path= f"{avatar_path}/latents.pt"
video_out_path = f"{avatar_path}/vid_output/"
mask_out_path =f"{avatar_path}/mask"
mask_coords_path =f"{avatar_path}/mask_coords.pkl"
avatar_info_path = f"{avatar_path}/avator_info.json"
avatar_info = {
    "avatar_id":avatar_id,
    "video_path":video_path,
    "bbox_shift":bbox_shift   
}

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


def prepare_material():
    if os.path.exists(avatar_path):
        print(f"avatar {avatar_id} already exists, skip preparation, please try to rename or delete the folder if you want to re-prepare")
        return
    print("preparing data materials ... ...")
    osmakedirs([avatar_path,full_imgs_path,mask_out_path,video_out_path])
    with open(avatar_info_path, "w") as f:
        json.dump(avatar_info, f)
        
    if os.path.isfile(video_path):
        video2imgs(video_path, full_imgs_path, ext = 'png')
    else:
        print(f"copy files in {video_path}")
        files = os.listdir(video_path)
        files.sort()
        files = [file for file in files if file.split(".")[-1]=="png"]
        for filename in files:
            shutil.copyfile(f"{video_path}/{filename}", f"{full_imgs_path}/{filename}")
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    
    print("extracting landmarks...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    input_latent_list = []
    idx = -1
    # maker if the bbox is not sufficient 
    coord_placeholder = (0.0,0.0,0.0,0.0)
    for bbox, frame in zip(coord_list, frame_list):
        idx = idx + 1
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)

    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    mask_coords_list_cycle = []
    mask_list_cycle = []

    for i,frame in enumerate(tqdm(frame_list_cycle)):
        cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png",frame)
        
        face_box = coord_list_cycle[i]
        mask,crop_box = get_image_prepare_material(frame,face_box)
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png",mask)
        mask_coords_list_cycle += [crop_box]
        mask_list_cycle.append(mask)
        
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)
        
    torch.save(input_latent_list_cycle, os.path.join(latents_out_path)) 

if __name__ == "__main__":
    prepare_material()
    print("preparation finished!")