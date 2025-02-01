import time
import os
import torch
import decord
import cv2
import numpy as np
from PIL import Image
from dataset import dataset_utils
from eval_utils.test_dataloader import load_query, load_clip, process_inputs
from einops import rearrange
from utils import vis_utils
import scipy
from scipy.signal import find_peaks, medfilt
from eval_utils.structures import BBox, ResponseTrack
import random

import timm
from torchvision import transforms
import clip
from sklearn.metrics.pairwise import cosine_similarity


SMOOTHING_SIGMA = 5
DISTANCE = 25
WIDTH = 3
PROMINENCE = 0.2
PEAK_SCORE_THRESHILD = 0.5  
PEAK_WINDWOW_RATIO = 0.5

PEAK_SCORE_THRESHOLD = 0.8
PEAK_WINDOW_THRESHOLD = 0.7
CLIP_WINDOW_THRESHOLD = 0.9

def extract_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    video = np.array(frames)
    return video


class Task:
    def __init__(self, config, annots, cuid=""):
        super().__init__()
        self.config = config
        self.annots = annots
        # Ensure that all annotations belong to the same clip
        if "clip_uid" in annots[0] and cuid == "":
            self.clip_uid = annots[0]["clip_uid"]
        else:
            self.clip_uid = cuid
        
        # for annot in self.annots:
        #     assert annot["clip_uid"] == clip_uid
        self.keys = [
            (annot["annotation_uid"], annot["query_set"])
            for annot in self.annots
        ]
        self.clip_dir = './DLCV_vq2d_data/clips'

        # self.model_clip, _ = clip.load("ViT-B/32", device="cuda")
        # self.model_clip = self.model_clip.eval()
        self.tfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def run(self, config, device, model):
        clip_uid = self.clip_uid

        # maybe some clip_uid is missing in original dataset
        if clip_uid is None:
            print(self.annots[0]["metadata"]["annotation_uid"])
            latest_bbox_format = [BBox(0, 0.0, 0.0, 0.0, 0.0)]
            all_pred_rts = {}
            for key, annot in zip(self.keys, self.annots):
                pred_rts = [ResponseTrack(latest_bbox_format, score=1.0)]
                all_pred_rts[key] = pred_rts
            return all_pred_rts
        ######################
        
        clip_path = os.path.join(self.clip_dir, clip_uid  + '.mp4')
        if not os.path.exists(clip_path):
            print(f"Clip {clip_uid} does not exist")
            return {}

        all_pred_rts = {}
        for key, annot in zip(self.keys, self.annots):
            annotation_uid = annot["annotation_uid"]
            query_set = annot["query_set"]
            annot_key = f"{clip_uid}_{annotation_uid}_{query_set}"
            # print(annot_key)
            query_frame = annot["query_frame"]
            object_title = annot["object_title"]
            visual_crop = annot["visual_crop"]
            # print(visual_crop)

            image_folder = "./DLCV_vq2d_data/images/val"
            query_image_path = os.path.join(image_folder, f"{clip_uid}/frame_{visual_crop['frame_number']+1:07d}.png")
            assert os.path.isfile(query_image_path), query_image_path

            query_image = Image.open(query_image_path)
            crop_box = (visual_crop['x'], visual_crop['y'], visual_crop['x']+visual_crop['width'], visual_crop['y']+visual_crop['height'])
            cropped_image = query_image.crop(crop_box)
            cropped_image.save(os.path.join('./cropped', f"{clip_uid}_frame_{visual_crop['frame_number']+1:07d}.png"))


            save_path = os.path.join(self.config.inference_cache_path, f'{annot_key}.pt')
            assert os.path.isfile(save_path), f"your path:{save_path}"
            cache = torch.load(save_path)
            ret_bboxes, ret_scores = cache['ret_bboxes'], torch.sigmoid(cache['ret_scores'])
            ret_bboxes = ret_bboxes.numpy()     # bbox in [N,4], original resolution, cv2 axis
            ret_scores = ret_scores.numpy()     # scores in [N]

            ret_scores_sm = ret_scores.copy()
            for i in range(1):
                ret_scores_sm = medfilt(ret_scores_sm, kernel_size=SMOOTHING_SIGMA)

            # only used for testing stAP with gt window 
            # gt_scores = np.zeros_like(ret_scores_sm)
            # len_clip = gt_scores.shape[0]
            # gt_rt_idx = [int(frame_it['frame_number']) for frame_it in annot['response_track']]
            # for frame_it in gt_rt_idx:
            #     gt_scores[min(frame_it, len_clip-1)] = random.uniform(0.6,1)
            # ret_scores_sm = gt_scores.copy()
                

            # Read video
            video_path = os.path.join("./DLCV_vq2d_data/clips/", f"{clip_uid}.mp4")
            video_frames = extract_frame(video_path)

            peaks, _ = find_peaks(ret_scores_sm)

            if len(peaks) == 0:
                print(ret_scores_sm)
            peaks = process_peaks(peaks, ret_scores_sm)

            recent_peak = None
            for peak in peaks[::-1]:
                recent_peak = peak
            '''
            peak_sim = []
            for peak in peaks:
                recent_peak = peak
                peak_bbox = ret_bboxes[peak]

                peak_frame = video_frames[peak]
                x1, y1, x2, y2 = peak_bbox
                # print(peak_bbox)
                predict_obj = peak_frame[int(y1):int(y2+1), int(x1):int(x2+1)]
                # cv2.imwrite(os.path.join("predict_obj",f"{clip_uid}_{annotation_uid}_{query_set}_{peak}.png"), predict_obj)
                predict_obj = Image.fromarray(predict_obj)
                predict_obj = self.tfm(predict_obj)

                text = clip.tokenize(f"A image of {object_title}").to("cuda")
                with torch.no_grad():
                    # image_features = model.encode_image(predict_obj.unsqueeze(0).to(device))
                    # text_features = model.encode_text(text)
                    logits_per_image, logits_per_text = model(predict_obj.unsqueeze(0).to(device), text)
                    # print(logits_per_image.softmax(dim=-1).cpu().item())
                    # print()
                    # peak_sim.append((peak, cosine_similarity(text_features.cpu().numpy(), image_features.cpu().numpy())[0][0]))
                    peak_sim.append((peak, logits_per_image.cpu().item()))
            '''
            
            print(f"GT: {annot['response_track'][0]['frame_number']}, {annot['response_track'][-1]['frame_number']}")
            # print(peak_sim)
            
# start 
            # recent_peak = 0
            text = clip.tokenize(f"A image of {object_title}").to("cuda")
            if recent_peak is not None:
                threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
                recent_bbox = ret_bboxes[recent_peak]
                peak_frame = video_frames[recent_peak]
                # x1, y1, x2, y2 = recent_bbox
                x1 = np.maximum(int(recent_bbox[0]),0)
                y1 = np.minimum(int(recent_bbox[1]),0)
                x2 = np.maximum(int(recent_bbox[2]),peak_frame.shape[1])
                y2 = np.minimum(int(recent_bbox[3]),peak_frame.shape[0])
                peak_obj = peak_frame[y1:y2, x1:x2]
                peak_obj = Image.fromarray(peak_obj)
                peak_obj = self.tfm(peak_obj)
                with torch.no_grad():
                    logits_per_image, _ = model(peak_obj.unsqueeze(0).to(device), text)
                threshold_clip = logits_per_image * CLIP_WINDOW_THRESHOLD
                latest_idx = [recent_peak]
                for idx in range(recent_peak, 0, -1):
                    recent_bbox = ret_bboxes[idx]
                    peak_frame = video_frames[idx]
                    x1 = np.maximum(int(recent_bbox[0]),0)
                    y1 = np.minimum(int(recent_bbox[1]),0)
                    x2 = np.maximum(int(recent_bbox[2]),peak_frame.shape[1])
                    y2 = np.minimum(int(recent_bbox[3]),peak_frame.shape[0])
                    peak_obj = peak_frame[y1:y2, x1:x2]
                    # x1, y1, x2, y2 = recent_bbox
                    # peak_obj = peak_frame[int(y1):int(y2), int(x1):int(x2)]
                    # peak_obj = peak_frame[int(y1+1):int(y2-1), int(x1+1):int(x2-1)]
                    peak_obj = Image.fromarray(peak_obj)
                    peak_obj = self.tfm(peak_obj)
                    with torch.no_grad():
                        logits_per_image, _ = model(peak_obj.unsqueeze(0).to(device), text)
                    if ret_scores_sm[idx] >= threshold and logits_per_image > threshold_clip:
                        latest_idx.append(idx)
                    else:
                        break
                for idx in range(recent_peak, query_frame-1):
                    recent_bbox = ret_bboxes[idx]
                    peak_frame = video_frames[idx]
                    x1 = np.maximum(int(recent_bbox[0]),0)
                    y1 = np.minimum(int(recent_bbox[1]),0)
                    x2 = np.maximum(int(recent_bbox[2]),peak_frame.shape[1])
                    y2 = np.minimum(int(recent_bbox[3]),peak_frame.shape[0])
                    peak_obj = peak_frame[y1:y2, x1:x2]
                    # x1, y1, x2, y2 = recent_bbox
                    # peak_obj = peak_frame[int(y1):int(y2+1), int(x1):int(x2+1)]
                    # peak_obj = peak_frame[int(y1+1):int(y2-1), int(x1+1):int(x2-1)]
                    peak_obj = Image.fromarray(peak_obj)
                    peak_obj = self.tfm(peak_obj)
                    with torch.no_grad():
                        logits_per_image, _ = model(peak_obj.unsqueeze(0).to(device), text)
                    if ret_scores_sm[idx] >= threshold and logits_per_image > threshold_clip:
                        latest_idx.append(idx)
                    else:
                        break
            else:
                latest_idx = [query_frame-2]
            
            latest_idx = sorted(list(set(latest_idx)))
            latest_bbox = ret_bboxes[latest_idx]    # [t,4]
            print(latest_idx)
            latest_bbox_format = []
            for (frame_bbox, fram_idx) in zip(latest_bbox, latest_idx):
                x1, y1, x2, y2 = frame_bbox
                bbox_format = BBox(fram_idx, x1, y1, x2, y2)
                latest_bbox_format.append(bbox_format)
            
            pred_rts = [ResponseTrack(latest_bbox_format, score=1.0)]
            all_pred_rts[key] = pred_rts
        
        return {clip_uid: all_pred_rts}


def process_peaks(peaks_idx, ret_scores_sm):
    '''process the peaks based on their scores'''
    num_frames = ret_scores_sm.shape[0]
    if len(peaks_idx) == 0:
        start_score, end_score = ret_scores_sm[0], ret_scores_sm[-1]
        if start_score > end_score:
            valid_peaks_idx = [0]
        else:
            valid_peaks_idx = [num_frames-1]
    else:
        peaks_score = ret_scores_sm[peaks_idx]
        largest_score = np.max(peaks_score)

        threshold = largest_score * PEAK_SCORE_THRESHOLD

        valid_peaks_idx_idx = np.where(peaks_score > threshold)[0]
        valid_peaks_idx = peaks_idx[valid_peaks_idx_idx]
    return valid_peaks_idx
