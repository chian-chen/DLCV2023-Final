python3 inference_predict.py --cfg ./config/val.yaml  
python3 inference_results_clip_2.py --cfg ./config/val.yaml 
python3 evaluation/evaluate_vq.py --pred-file val_results.json --gt-file ./DLCV_vq2d_data/vq_val.json