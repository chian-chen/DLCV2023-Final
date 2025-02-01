import json
import os

def get_object(jf):
    obj_set = set()
    for v in jf.values():
        assert isinstance(v["annotations"], list)
        for a in v["annotations"]:
            for q in a["query_sets"].values():
                obj_set.add(q["object_title"])
    return obj_set

DATASET_PATH = "./DLCV_vq2d_data"
TRAIN_JSON = os.path.join(DATASET_PATH, "vq_train.json")
VAL_JSON = os.path.join(DATASET_PATH, "vq_val.json")
TEST_JSON = os.path.join(DATASET_PATH, "vq_test_unannotated.json")


with open(TRAIN_JSON, "r") as f:
    train_set = json.load(f)
with open(VAL_JSON, "r") as f:
    val_set = json.load(f)
with open(TEST_JSON, "r") as f:
    test_set = json.load(f)

train_obj_set = get_object(train_set)
val_obj_set = get_object(val_set)
test_obj_set = get_object(test_set)


diff_train_val_set = val_obj_set.difference(train_obj_set)
diff_train_test_set = test_obj_set.difference(train_obj_set)
print("diff from train and val",len(diff_train_val_set))
print("diff from train and test",len(diff_train_test_set))
print(f"train {len(train_obj_set)} | val {len(val_obj_set)} | test {len(test_obj_set)}")

union_train_val_set = val_obj_set | train_obj_set
diff_union_test_set = test_obj_set - union_train_val_set
intersection = val_obj_set & test_obj_set & train_obj_set
print("diff from union and test set", len(diff_union_test_set))
print("intersection", len(intersection))
