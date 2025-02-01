import json
import os
import torch

## json
with open('val_results.json', 'r') as file:
    data = json.load(file)

with open('DLCV_vq2d_data/vq_val.json', 'r') as info_file:
    clip_info = json.load(info_file)

## 
count = 0
for uuid, prediction_data in data.items():
    for prediction in prediction_data['predictions']:
        query_sets = prediction['query_sets']

        for key, frame_data in query_sets.items():
            bboxes = frame_data['bboxes']
            flagged = False

            for i in range(len(bboxes) - 1):
                bbox_current = bboxes[i]
                bbox_next = bboxes[i + 1]

                center_current = ((bbox_current['x1'] + bbox_current['x2']) / 2, (bbox_current['y1'] + bbox_current['y2']) / 2)
                center_next = ((bbox_next['x1'] + bbox_next['x2']) / 2, (bbox_next['y1'] + bbox_next['y2']) / 2)

                delta_x = abs(center_current[0] - center_next[0])
                delta_y = abs(center_current[1] - center_next[1])

                image_width = clip_info[uuid]['annotations'][0]['query_sets']['2']['visual_crop']['original_width']
                image_height = clip_info[uuid]['annotations'][0]['query_sets']['2']['visual_crop']['original_height']

                ## logic
                if delta_x > image_width * 0.75 or delta_y > image_height * 0.75:
                    flagged = True
                    print(uuid)
                    count += 1
                    break

            if flagged:
                query_sets[key]['bboxes'] = bboxes[:i + 1]

print(count)
with open('modified_val_results.json', 'w') as file:
    json.dump(data, file)
