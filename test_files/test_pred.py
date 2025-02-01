import torch
import os, sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np
import json
from eval_utils import eval_utils

id = "044ffbef-bd17-43ef-8fa9-616a8c484b25"
# from gt
annotation_path = "./DLCV_vq2d_data/vq_val.json"
with open(annotation_path) as fp:
    annotations = json.load(fp)
clipwise_annotations_list = eval_utils.custom_convert_annotations_to_clipwise_list(annotations)
gt_dict = clipwise_annotations_list[id]

aid, qset = 0,"1"
for i in gt_dict:
    if i["query_set"] == qset and i["annotation_uid"] == aid:
        gt = i
t_query = np.arange(gt["query_frame"])
gt_curve = np.zeros(gt["query_frame"])
print("query len",gt["query_frame"])

for i in gt["response_track"]:
    gt_curve[i["frame_number"]] = 1

output = torch.load(f"debug/044ffbef-bd17-43ef-8fa9-616a8c484b25_{aid}_{qset}.pt")
pred = output["ret_scores"]
exist_pred = output["exist_prob"]
print(pred.shape, exist_pred.shape)
t = np.arange(min(pred.shape[0], exist_pred.shape[0]))
plt.plot(t,gt_curve,color='r',marker=None,label="gt")
plt.plot(t,torch.sigmoid(pred),color='b',marker=None, label="pred by box")
plt.plot(t,torch.sigmoid(exist_pred[:len(t)]),color='g',marker=None, label="pred by head")
plt.legend()
plt.savefig(f"./debug/{id}_{aid}_{qset}_curve.jpg")