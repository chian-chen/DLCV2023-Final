# DLCV Final Project ( Visual Queries 2D Localization Task )

# Intuition

We provide a 'DLCV_final.pdf' file, which outlines our conceptual framework and offers detailed insights into our error studies and implementation specifics. Our work primarily consists of the following steps:

1. Understand and replicate the results of the state-of-the-art work 'VQLoc.'
2. Start improving temporal resolution, based on the observation that the difference between stAP and tAP is minimal.
3. After visualizing error cases and conducting research, propose post-processing based on the use of CLIP and the continuity of object.
4. After obtaining results, attempt to fine-tune settings.


# installation
```shell script=
sh installation.sh
```

# How to run our code

**Prepare dataset**
```shell script=
# download dataset
sh get_dataset.sh
# prepare dataset for training and inference
sh prepare_dataset.sh
```
Note: if "sh prepare_dataset.sh" failed, delete all files in "./DLCV_vq2d_data/annot/" and do it again after debugging


**Train**

```shell script=
# get pretrained weight
sh get_weight.sh
sh train.sh
```

**Inference by Pretrained Weight**
```shell script=
# get pretrained weight
sh get_weight.sh
sh test_set_inference.sh
```

**Post Processing**
```shell script=
python3 post_processing.py
# CLIP post processing (Not used in final submission)
sh inference_predict_clip.sh
```

# Results (on val set)

Baseline:

|  stAP   | tAP    |
|  ----   | ----   |
| 0.2306  | 0.3039 |


Temporal resolution improvement:

|  stAP   | tAP    |
|  ----   | ----   |
| 0.2742  | 0.3595 |

Post processing (CLIP NOT included):

|  stAP   | tAP    |
|  ----   | ----   |
| 0.2284  | 0.3049 |

Post processing (CLIP ONLY):

|  stAP   | tAP    |
|  ----   | ----   |
| 0.0563  | 0.0823 |

---

# Usage
To start working on this final project, you should clone this repository into your local machine by the following command:

    git clone https://github.com/ntudlcv/DLCV-Fall-2023-Final-2-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1TsR0l84wWNNWH7HaV-FEPFudr3o9mVz29LZQhFO22Vk/edit?usp=sharing) to view the slides of Final Project - Visual Queries 2D Localization Task. **The introduction video for final project can be accessed in the slides.**

# Visualization
We provide the code for visualizing your predicted bounding box on each frame. You can run the code by the following command:

    python3 visualize_annotations.py --annot-path <annot-path> --clips-root <clips-root> --vis-save-root <vis-save-root>

Note that you should replace `<annot-path>`, `<clips-root>`, and `<vis-save-root>` with your annotation file (e.g. `vq_val.json`), the folder contains all clips, and the output folder of the visualization results, respectively.

# Evaluation
We also provide the evaluation for you to check the performance (stAP) on validation set. You can run the code by the following command:

    cd evaluation/
    python3 evaluate_vq.py --gt-file <gt-file> --pred-file <pred-file>

Note that you should replace `<gt-file>` with your val annotation file (e.g. `vq_val.json`) and replace `<pred-file>` with your output prediction file (e.g. `pred.json`)  
