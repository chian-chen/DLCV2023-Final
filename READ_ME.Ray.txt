Prepare dataset:
    1. download dataset from TA first
        - run "get_dataset.sh"
    2. execute prepare_dataset.sh
        - if some module needed, install by yourself cuz this is from previous repo QQ

Evaluate:
    1. download pretrained weight
        - go to github
        - or run "get_weight.sh"
    2. run "test_set_inference.sh" for test set, or "val_set_inference.sh" for val set

Train:
    1. run "train.sh"
    - if wanna modify model, go to ./model/corr_clip_spatial_transformer2_anchor_2heads_hnm.py
    - if wanna modify loss, go to ./utils/loss_utils.py
    - if wanna change config, go to ./config/train.yaml


To ZhenXun:
    1. it is trainable now, just run train.sh then you could go to sleep
    2. I havent not modified "./eval_utils/task_inference_predict.py" yet. when inference there may be some error when run inference_predict.py
    3. for inference_results.py, just uncomment line 22 and comment 23, then it could work.