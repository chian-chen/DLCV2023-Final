# from dataset.b import QueryVideoDataset
import sys
sys.path.append(".")
from config.config import config, update_config
from dataset import dataset_utils   

arg_config = "./config/train.yaml"
update_config(arg_config)
train_data = dataset_utils.get_dataset(config, split='train')
data = train_data[0]
print(len(data["clip"]))
print(data["clip_with_bbox"])
print(data["before_query"])
