"""
Script to extract images from a video
"""
import argparse
import collections
import json
import multiprocessing as mp
import os

import imageio
import pims
import tqdm
from constants import CLIP_NAME_PATTERN
from utils import get_image_name_from_clip_uid
# from vq2d.baselines.utils import get_image_name_from_clip_uid

def save_video_frames(path, frames_to_save):
    # video_md = read_video_md(path)
    frames_to_save_dict = collections.defaultdict(list)
    for fs in frames_to_save:
        frames_to_save_dict[fs["video_fno"]].append(fs["save_path"])
    
    reader = pims.Video(path)
    
    for data in frames_to_save:
        fno = data["video_fno"]
        _path = data["save_path"]
        f = reader[fno]

        if not os.path.isfile(_path) or os.path.getsize(_path) == 0:
            imageio.imwrite(_path, f)    

def frames_to_select(
    start_frame: int,
    end_frame: int,
    original_fps: int,
    new_fps: int,
):
    # ensure the new fps is divisible by the old
    assert original_fps % new_fps == 0

    # check some obvious things
    assert end_frame >= start_frame

    num_frames = end_frame - start_frame + 1
    skip_number = original_fps // new_fps
    for i in range(0, num_frames, skip_number):
        yield i + start_frame


# one video in, and save the images
def extract_clip_frame_nos(video_uid, clip_annotation, save_root):
    """
    Extracts frame numbers corresponding to the VQ annotation for a given clip

    Args:
        video_md - a dictionary of video metadata
        clip_annotation - a clip annotation from the VQ task export
        save_root - path to save extracted images
    """
    """
    DLCV DATA:
        video, clip_fps, fps, video_start_frame, video_end_frame
        just do the for loop
    """
    frames_to_save = []
    for annotation in clip_annotation["annotations"]:
        for qset_id, qset in annotation["query_sets"].items():
            if not qset["is_valid"]:
                continue
            q_fno = qset["query_frame"] # query frame
            vc_fno = qset["visual_crop"]["frame_number"] # query object's frame
            rt_fnos = [rf["frame_number"] for rf in qset["response_track"]]
            # also add negative frames
            rt_fnos = sorted(
                rt_fnos
            )
            rt_dur = rt_fnos[-1] - rt_fnos[0] + 1   # duration

            # negative frames, at most the same length with positive frames
            rtn_fnos = [rt_fno+rt_dur for rt_fno in rt_fnos if rt_fno+rt_dur<q_fno]

            # all frames
            all_fnos = [vc_fno] + rt_fnos + rtn_fnos
            for fno in all_fnos:
                path = os.path.join(save_root, get_image_name_from_clip_uid(video_uid, fno))
                if os.path.isfile(path) and os.path.getsize(path)>0:
                    continue
                frames_to_save.append(
                    {"video_fno": fno, "save_path": path}
                )
    return frames_to_save


def batchify_video_uids(video_uids, batch_size):
    video_uid_batches = []
    nbatches = len(video_uids) // batch_size
    if batch_size * nbatches < len(video_uids):
        nbatches += 1
    for batch_ix in range(nbatches):
        video_uid_batches.append(
            video_uids[batch_ix * batch_size : (batch_ix + 1) * batch_size]
        )
    return video_uid_batches


def video_to_image_fn(inputs):
    data, args = inputs
    video_uid = data["video_uid"]
    video_data = data["data"]

    # Extract frames for a specific video_uid
    video_path = os.path.join(args.ego4d_videos_root, video_uid + ".mp4")
    if not os.path.isfile(video_path):
        print(f"Missing video {video_path}")
        return None

    # Get list of frames to save for annotated clips
    frame_nos_to_save = []

    os.makedirs(os.path.join(args.save_root, video_uid), exist_ok=True)

    frame_nos_to_save = extract_clip_frame_nos(video_uid, video_data, args.save_root)
    
    if len(frame_nos_to_save) == 0:
        # print(f"=========> No valid frames to read for {video_uid}!")
        return None

    save_video_frames(video_path, frame_nos_to_save)
    print(f"=========> Found valid frames to read for {video_uid}!")
    


def main(args):
    # Load annotations
    annotation_export = []
    
    with open(args.annot_paths, "r") as f:
        annot_dict = json.load(f)
    
    video_uids = sorted([key for key in annot_dict.keys()])

    os.makedirs(args.save_root, exist_ok=True)
    
    # do certain batch
    if args.video_batch_idx >= 0:
        video_uid_batches = batchify_video_uids(video_uids, args.video_batch_size)
        video_uids = video_uid_batches[args.video_batch_idx]
        print(f"===> Processing video_uids: {video_uids}")
    
    
    # Get annotations corresponding to video_uids
    annotation_export = [{"video_uid": v, "data":annot_dict[v]} for v in video_uids]

    pool = mp.Pool(args.num_workers)
    inputs = [(video_data, args) for video_data in annotation_export]
    # _inputs = inputs[:2]
    _ = list(
        tqdm.tqdm(
            pool.imap_unordered(video_to_image_fn, inputs),
            total=len(inputs),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-batch-idx", type=int, default=-1)
    parser.add_argument("--annot-paths", type=str, default="./DLCV_vq2d_data/vq_train.json")
    parser.add_argument("--save-root", type=str, default="./DLCV_vq2d_data/images/train")
    parser.add_argument("--ego4d-videos-root", type=str, default="./DLCV_vq2d_data/clips")
    parser.add_argument("--video-batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=10)
    args = parser.parse_args()

    main(args)