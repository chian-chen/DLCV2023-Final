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


SMOOTHING_SIGMA = 5

##### unused ######
DISTANCE = 25
WIDTH = 3
PROMINENCE = 0.2
PEAK_SCORE_THRESHILD = 0.5  
PEAK_WINDWOW_RATIO = 0.5
###################

PEAK_ABS_THRESHOLD = 0.6
PEAK_SCORE_THRESHOLD = 0.8      # TODO: modify this to get best AP
PEAK_WINDOW_THRESHOLD = 0.7     
EXIST_THESHOLD = 10

# TODO: read below how to find peak
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

    def run(self, config, device):
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
        case = [0,0,0]
        for key, annot in zip(self.keys, self.annots):
            annotation_uid = annot["annotation_uid"] # aid       
            query_set = annot["query_set"]  # qset
            annot_key = f"{clip_uid}_{annotation_uid}_{query_set}"
            query_frame = annot["query_frame"]
            visual_crop = annot["visual_crop"]
            save_path = os.path.join(self.config.inference_cache_path, f'{annot_key}.pt')
            assert os.path.isfile(save_path), f"your path:{save_path}"
            cache = torch.load(save_path)
            ret_bboxes, ret_scores = cache['ret_bboxes'], torch.sigmoid(cache['ret_scores'])
            ret_bboxes = ret_bboxes.numpy()     # bbox in [N,4], original resolution, cv2 axis
            ret_scores = ret_scores.numpy()     # scores in [N]

            exist_pred, exist_pred_sigmoid = cache["exist_prob"], torch.sigmoid(cache["exist_prob"])
            exist_pred = exist_pred.numpy()
            exist_pred_sigmoid = exist_pred_sigmoid.numpy()

            ret_scores_sm = ret_scores.copy()
            ret_scores_sm = medfilt(ret_scores_sm, kernel_size=SMOOTHING_SIGMA)

            peaks, _ = find_peaks(ret_scores_sm)

            exist_scores = exist_pred   # no sigmoid
            exist_peaks_idx, _ = find_peaks(exist_scores)
            exist_peak_scores = exist_scores[exist_peaks_idx]
            
            exist_threshold = min(max(exist_peak_scores) * 0.10, EXIST_THESHOLD) if len(exist_peak_scores) != 0 else EXIST_THESHOLD
            
            # valid_exist_peak_idx_idx = np.where(exist_pred_sigmoid[exist_peaks_idx] > 0.8)
            valid_exist_peak_idx_idx = np.where(exist_pred[exist_peaks_idx] > exist_threshold)
            
            valid_exist_peak_idx = exist_peaks_idx[valid_exist_peak_idx_idx]
            corr_ret_scores_sm = ret_scores_sm[valid_exist_peak_idx]
            if len(peaks) == 0:
                ret_score_th = PEAK_ABS_THRESHOLD
            else:
                ret_score_th = min(max(ret_scores_sm[peaks]) * PEAK_SCORE_THRESHOLD,  PEAK_ABS_THRESHOLD)
            
            corr_ret_scores_sm_th_idx = np.where(corr_ret_scores_sm > ret_score_th)
            # the idx that over two threshold
            valid_peak_2th_idx = valid_exist_peak_idx[corr_ret_scores_sm_th_idx]
            
            # find recent peak
            if len(valid_peak_2th_idx) > 0:
                # priority 1
                case[0] += 1
                ret_candidate = valid_peak_2th_idx[-1]
                prev_peak = 0
                nearest_large_peak = 0
                nearest_small_peak = 0
                for i in peaks:
                    if ret_candidate <= i:
                        nearest_large_peak = i
                        nearest_small_peak = prev_peak
                        break
                    prev_peak = i

                if nearest_large_peak == 0:
                    recent_peak = peaks[-1] if len(peaks) != 0 else ret_candidate 
                else:
                    peak_threshold = 0.6
                    scores_peak_interval = ret_scores_sm[nearest_small_peak:nearest_large_peak+1]
                    idx_under_th = np.where(scores_peak_interval < ret_scores_sm[ret_candidate]*peak_threshold)[0]

                    if len(idx_under_th) == 0:
                        if ret_scores_sm[nearest_large_peak] > ret_scores_sm[nearest_small_peak]:
                            recent_peak = nearest_large_peak
                        else:
                            recent_peak = nearest_small_peak
                    else:
                        first_idx_under_th = idx_under_th[0]
                        if first_idx_under_th > (ret_candidate - nearest_small_peak):
                            recent_peak = nearest_small_peak
                        else:
                            recent_peak = nearest_large_peak
                threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
                latest_idx = get_range_of_latest_appearence(recent_peak, ret_scores_sm, threshold)
            # elif len(valid_exist_peak_idx) > 0:
            #     # priority 2
            #     case[1] += 1
            #     exist_candidate = valid_exist_peak_idx[-1]
            #     latest_idx = get_range_of_latest_appearence(exist_candidate, exist_scores, 0)
            else:
                # priority 3
                case[2] += 1
                if len(peaks) == 0:
                    print(ret_scores_sm)
                peaks = process_peaks(peaks, ret_scores_sm)
                recent_peak = peaks[-1] if len(peaks) > 0 else None
                if recent_peak is not None:
                    threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
                    latest_idx = get_range_of_latest_appearence(recent_peak, ret_scores_sm, threshold)
                else:
                    print(f"clip uid: {cid}\nannotation uid: {aid}\nquery set: {qset}")
                    latest_idx = [query_frame-2]
                    # answer = ret_scores_sm[latest_idx]
            
            latest_idx = sorted(list(set(latest_idx)))
            latest_bbox = ret_bboxes[latest_idx]    # [t,4]
            
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

        threshold = min(largest_score * PEAK_SCORE_THRESHOLD, PEAK_ABS_THRESHOLD)

        valid_peaks_idx_idx = np.where(peaks_score > threshold)[0]
        valid_peaks_idx = peaks_idx[valid_peaks_idx_idx]
    return valid_peaks_idx

def get_range_of_latest_appearence(recent_peak, base_scores, threshold):
    latest_idx = [recent_peak]

    for idx in range(recent_peak,0,-1):
        if base_scores[idx] >= threshold:
            latest_idx.append(idx)
        else:
            break
    query_frame = len(base_scores)
    for idx in range(recent_peak, query_frame-1):
        if base_scores[idx] >= threshold:
            latest_idx.append(idx)
        else:
            break
    return latest_idx
    # answer = np.zeros(query_frame)
    # answer[latest_idx] = 1
    # return answer