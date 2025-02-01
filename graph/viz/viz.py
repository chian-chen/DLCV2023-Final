import os
import cv2
import numpy as np
import argparse
import json


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def draw_bounding_boxes(frame, boxes, color=(0, 0, 255)):
    """
    Draw bounding boxes on a frame.
    """
    for box in boxes:
        x1, y1, width, height = box['x'], box['y'], box['width'], box['height']
        x2, y2 = x1 + width, y1 + height
        draw_frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return draw_frame

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

def crop_boxes(frame, box):
    x1, y1, width, height = box['x'], box['y'], box['width'], box['height']
    x2, y2 = x1 + width, y1 + height
    cropped_image = frame[y1:y2, x1:x2]
    return cropped_image

# ============================================================================================================

def draw_pred(pred, args):
    """
    Process the data to extract frame information from video files and draw bounding boxes.
    """
    # Process each video
    for video_id in pred.keys():
        video_path = os.path.join(args.to_video_clips, f'{video_id}.mp4')  # Replace with actual video file path
        video = extract_frame(video_path)
        os.makedirs(f'./viz/results/{video_id}', exist_ok=True)

        # Retrieve predictions and ground truth for each video
        predictions = pred[video_id]['predictions']

        # Process each frame
        for j, prediction in enumerate(predictions):

            for i, query_set in zip(prediction['query_sets'].keys(), prediction['query_sets'].values()):

                for bbox in query_set['bboxes']:
                    frame_number = bbox['fno']
                    pred_box = {'x': bbox['x1'], 'y': bbox['y1'], 'width': bbox['x2'] - bbox['x1'], 'height': bbox['y2'] - bbox['y1']}

                    # Extract frame from video
                    frame = video[frame_number]
                   
                    if frame is not None:
                        draw_frame = draw_bounding_boxes(frame, [pred_box], color=(255, 0, 0))
                        # Save frame
                        cv2.imwrite(os.path.join(f'./viz/results/{video_id}', f'{j}_{i}_{frame_number:04d}.png'), draw_frame)


def draw_gt(gt, args):

    for video_id in gt.keys():
        video_path = os.path.join(args.to_video_clips, f'{video_id}.mp4')  # Replace with actual video file path
        video = extract_frame(video_path)
        output_path = f'./viz/results/{video_id}'
        os.makedirs(output_path, exist_ok=True)


        # gts = gt[video_id]['annotations'][0]['query_sets']
        anno = gt[video_id]['annotations']

        for j, gts in enumerate(anno):
            gts = gts['query_sets']
            
            for i in range(len(gts)):
                response_track = gts[f'{i+1}']['response_track']

                query_title = gts[f'{i+1}']['object_title']
                query_crop = gts[f'{i+1}']['visual_crop']

                frame_number = query_crop['frame_number']
                pred_box = {'x': query_crop['x'], 'y': query_crop['y'], 'width': query_crop['width'], 'height': query_crop['height']}

                query_path = os.path.join(output_path, f'{j}_{i + 1}_{query_title}.png')
                query_img = crop_boxes(video[frame_number], pred_box)
                h, w = query_img.shape[:2]

                for bbox in response_track:
                    frame_number = bbox['frame_number']
                    pred_box = {'x': bbox['x'], 'y': bbox['y'], 'width': bbox['width'], 'height': bbox['height']}
                    image_path = os.path.join(output_path, f'{j}_{i + 1}_{frame_number:04d}.png')

                    if os.path.exists(image_path):
                        draw_frame = cv2.imread(image_path)
                    else:
                        draw_frame = video[frame_number]

                    
                    draw_frame = draw_bounding_boxes(draw_frame, [pred_box], color=(0, 0, 255))
                    # Save frame
                    cv2.imwrite(image_path, draw_frame)

            
                for filename in os.listdir(output_path):
                    if filename.startswith(f'{j}_{i + 1}'):
                        file_path = os.path.join(output_path, filename)
                        im = cv2.imread(file_path)
                        height, _ = im.shape[:2]
                        
                        padding_vertical = (height - h) // 2
                        query_with_padding = cv2.copyMakeBorder(query_img, padding_vertical, height - h - padding_vertical, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                        merged_image = cv2.hconcat([im, query_with_padding])
                        cv2.imwrite(file_path, merged_image)
                cv2.imwrite(query_path, query_img)


# ============================================================================================================
                        
def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--to_video_clips', type=str,
                        default='./DLCV_vq2d_data/clips/')
    parser.add_argument('--to_gt_json', type=str,
                        default='./DLCV_vq2d_data/vq_val.json')
    parser.add_argument('--to_pred_json', type=str,
                        default='./val_results.json')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    gt = load_json(args.to_gt_json)
    pred = load_json(args.to_pred_json)
    assert len(gt) == len(pred), "File length Error"

    draw_pred(pred, args)
    draw_gt(gt, args)

    