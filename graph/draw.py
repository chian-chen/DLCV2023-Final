import argparse
import os, sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np
import json
from eval_utils import eval_utils
import torch
from scipy.signal import medfilt, find_peaks

# PARAMETER
SMOOTHING_SIGMA = 5
PEAK_SCORE_THRESHOLD = 0.8      # TODO: modify this to get best AP
PEAK_WINDOW_THRESHOLD = 0.7 
PEAK_ABS_THRESHOLD = 0.6
EXIST_THESHOLD = 10

def parse_args():
    parser = argparse.ArgumentParser(description="draw graph")
    parser.add_argument(
        '--pt-dir', help="the pt files of prediction", default="output/ego4d_vq2d/val/validate/inference_cache_eval_from_spatial"
    )
    parser.add_argument(
        "--clip-id", help="the clip id that to draw", default=None
    )
    parser.add_argument(
        "--output", help="output direction", default="./graph/eval"
    )
    args, rest = parser.parse_known_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    return args

def main():
    args = parse_args()
    print(args)
    if args.clip_id is None:
        all_clips = os.listdir(args.pt_dir)
        clip_ids = set([clip.split("_")[0] for clip in all_clips])
        clip_ids = sorted(list(clip_ids))
        # clip_id = ["1aa16048-822d-40c4-9ba7-e1a1f150021e"]
        clip_id = clip_ids
        print(clip_id)
    else:
        clip_id = [args.clip_id]

    annotation_path = "./DLCV_vq2d_data/vq_test_unannotated.json"
    with open(annotation_path) as fp:
        annotations = json.load(fp)
    clipwise_annotations_list = eval_utils.custom_convert_annotations_to_clipwise_list(annotations)
    
    queries = [query for query in os.listdir(args.pt_dir) if query.split("_")[0] in clip_id ]
    
    
    for query in queries:
        
        cid, aid, qset = query.split("_")
        aid = int(aid)
        qset = qset.split(".")[0]
        gt_dict = clipwise_annotations_list[cid]

        for i in gt_dict:
            if i["query_set"] == qset and i["annotation_uid"] == aid:
                gt = i
        
        t = np.arange(gt["query_frame"])
        gt_curve = np.zeros(gt["query_frame"])
        query_frame = gt["query_frame"]
        if "response_track" in gt:
            for i in gt["response_track"]:
                if i["frame_number"] >= len(gt_curve):
                    break
                gt_curve[i["frame_number"]] = 1
        
        print(f"clip uid: {cid}\nannotation uid: {aid}\nquery set: {qset}")

        preds = torch.load(os.path.join(args.pt_dir, query))
        ret_scores = torch.sigmoid(preds["ret_scores"])
        ret_scores_sm = medfilt(ret_scores, SMOOTHING_SIGMA)
        exist_pred = preds["exist_prob"]
        
        # get peaks

        peaks, _ = find_peaks(ret_scores_sm)
        # exist_scores = torch.sigmoid(exist_pred)
        exist_scores = exist_pred
        exist_scores_sm = medfilt(exist_scores, SMOOTHING_SIGMA)
        exist_peaks_idx, _ = find_peaks(exist_scores)
        exist_peak_scores = exist_scores[exist_peaks_idx]
        exist_threshold = min(max(exist_peak_scores) * 0.10, EXIST_THESHOLD) if len(exist_peak_scores) != 0 else EXIST_THESHOLD

        # valid_exist_peak_idx_idx = np.where(torch.sigmoid(exist_peak_scores) > 0.7)
        valid_exist_peak_idx_idx = np.where(exist_peak_scores > exist_threshold)
        valid_exist_peak_idx = exist_peaks_idx[valid_exist_peak_idx_idx]
        
        corr_ret_scores_sm = ret_scores_sm[valid_exist_peak_idx]
        
        if len(peaks) == 0:
            ret_score_th = PEAK_ABS_THRESHOLD
        else:
            ret_score_th = min(max(ret_scores_sm[peaks]) * PEAK_SCORE_THRESHOLD,  PEAK_ABS_THRESHOLD)
        
        corr_ret_scores_sm_th_idx = np.where(corr_ret_scores_sm > ret_score_th)
        # the idx that over two thresholds
        valid_peak_2th_idx = valid_exist_peak_idx[corr_ret_scores_sm_th_idx]
        

        # find recent 
        if len(valid_peak_2th_idx) > 0:
            # priority 1: over two threshold
            print("priority 1")
            ret_candidate = valid_peak_2th_idx[-1]
            # print(f"candidate:{ret_candidate}")
            prev_peak = 0
            nearest_large_peak = 0
            nearest_small_peak = 0
            for i in peaks:
                if ret_candidate <= i:
                    nearest_large_peak = i
                    nearest_small_peak = prev_peak
                    break    
                prev_peak = i
            print(ret_candidate, nearest_large_peak, nearest_small_peak)
            if nearest_large_peak == 0:
                recent_peak = peaks[-1] if len(peaks) != 0 else ret_candidate 
            else:
                peak_threshold = 0.8     # MODIFY: FINETUNE IT
                scores_peak_interval = ret_scores_sm[nearest_small_peak:nearest_large_peak+1]
                idx_under_th = np.where(scores_peak_interval < ret_scores_sm[ret_candidate]*peak_threshold)[0]

                if len(idx_under_th) == 0:
                    if ret_scores_sm[nearest_large_peak] > ret_scores_sm[nearest_small_peak]:
                        recent_peak = nearest_large_peak
                    else:
                        recent_peak = nearest_small_peak
                else:
                    first_idx_under_th = idx_under_th[0]
                    print(first_idx_under_th, ret_candidate - nearest_small_peak)
                    if first_idx_under_th > (ret_candidate - nearest_small_peak):
                        recent_peak = nearest_small_peak
                    else:
                        recent_peak = nearest_large_peak

            print(f"priority 1, recent_peak: {recent_peak}")
            threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
            answer = get_range_of_latest_appearence(recent_peak, ret_scores_sm, threshold)

        elif len(valid_exist_peak_idx) > 0:
            print("priority 2")
            
            # priority 2: ret_scores not over threshold, but exist_peak over it
            exist_candidate = valid_exist_peak_idx[-1]
            answer = get_range_of_latest_appearence(exist_candidate, exist_scores, 0)
        else:
            # priority 3: the origin algo
            print("priority 3")
            if len(peaks) == 0:
                print(ret_scores_sm)
            peaks, th2 = process_peaks(peaks, ret_scores_sm)
            recent_peak = peaks[-1] if len(peaks) > 0 else None
            if recent_peak is not None:
                threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
                answer = get_range_of_latest_appearence(recent_peak, ret_scores_sm, threshold)
            else:
                # print(f"clip uid: {cid}\nannotation uid: {aid}\nquery set: {qset}")
                latest_idx = [query_frame-2]
                answer = ret_scores_sm[latest_idx]
            

        ##### answer by author #####
        if len(peaks) == 0:
            print(ret_scores_sm)
        peaks, th2 = process_peaks(peaks, ret_scores_sm)

        recent_peak = None
        for peak in peaks[::-1]:    # what the fuck is this??
            recent_peak = peak
            break

        if recent_peak is not None:
            threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
            latest_idx = [recent_peak]
            for idx in range(recent_peak, 0, -1):
                if ret_scores_sm[idx] >= threshold:
                    latest_idx.append(idx)
                else:
                    break
            for idx in range(recent_peak, query_frame-1):
                if ret_scores_sm[idx] >= threshold:
                    latest_idx.append(idx)
                else:
                    break
        else:
            print(f"clip uid: {cid}\nannotation uid: {aid}\nquery set: {qset}")
            latest_idx = [query_frame-2]
        answer_ori = np.zeros(query_frame)
        answer_ori[latest_idx] = 1
        ##############################

        

        # assert len(t) == len(ret_scores_sm), f'query_frame: {query_frame}, len(t): {len(t)}, len(score): {len(ret_scores_sm)}'
        if len(t) > len(ret_scores):
            t = t[:len(ret_scores)]
        # if len(t) > 300:
        #     t = t[-300:]
        # fig, axes = plt.subplots(1,2)
        # axes = axes.flatten()
        plt.plot(t,torch.sigmoid(exist_scores[t]),color='g',marker=None, label="pred by head")
        # plt.plot(t,gt_curve[t]*1.2,color='r',marker=None,label="gt")
        plt.plot(t,answer[t]*1.1, color="orange",marker=None, label="answer by me")
        plt.plot(t,answer_ori[t]*1.3, color="purple",marker=None, label="answer origin")
        plt.hlines(ret_score_th, t[0], query_frame-1, color="c")
        plt.plot(t,ret_scores_sm[t],color='b',marker=None, label="pred by box")
        plt.legend()
        # axes[1].plot(t,exist_scores[t], color='g', marker=None, label="pred by head")
        # axes[1].hlines(exist_threshold, t[0], query_frame-1)
        # axes[0].hlines(th2, t[0], query_frame-1, color="purple")
        plt.savefig(os.path.join(args.output,f"{cid}_{aid}_{qset}_curve.jpg"))
        plt.close()

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
    
    answer = np.zeros(query_frame)
    answer[latest_idx] = 1
    return answer

def process_peaks(peaks_idx, ret_scores_sm):
    '''process the peaks based on their scores'''

    num_frames = ret_scores_sm.shape[0]
    if len(peaks_idx) == 0:
        start_score, end_score = ret_scores_sm[0], ret_scores_sm[-1]
        if start_score > end_score:
            valid_peaks_idx = [0]
        else:
            valid_peaks_idx = [num_frames-1]
        threshold = 0
    else:
        peaks_score = ret_scores_sm[peaks_idx]
        largest_score = np.max(peaks_score)

        # threshold = min(largest_score * PEAK_SCORE_THRESHOLD, PEAK_ABS_THRESHOLD)
        threshold = largest_score * PEAK_SCORE_THRESHOLD
        valid_peaks_idx_idx = np.where(peaks_score > threshold)[0]
        valid_peaks_idx = np.array(peaks_idx)[valid_peaks_idx_idx]
    return valid_peaks_idx, threshold

if __name__ == "__main__":
    main()