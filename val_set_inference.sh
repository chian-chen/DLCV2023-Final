python3 inference_predict.py --cfg ./config/val.yaml  
python3 inference_results.py --cfg ./config/val.yaml --inference_path output/ego4d_vq2d/val/validate/inference_cache_val
python3 evaluation/evaluate_vq.py --pred-file val_results.json --gt-file ./DLCV_vq2d_data/vq_val.json