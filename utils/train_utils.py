import os
import pprint
import random
import numpy as np
import torch
import warnings

def get_optimizer(config, model):
    if config.model.fix_backbone:
        if config.train.fix_pretrained:
            for name, param in model.named_parameters():
                param.requires_grad = False
                module = name.split('.')[0]
                if module in config.train.trainable_module:
                    param.requires_grad = True
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(params,
                                lr=config.train.lr,
                                weight_decay=config.train.weight_decay)
    else:
        backbone_param = list(filter(lambda kv: 'backbone' in kv[0], model.named_parameters()))
        other_param = list(filter(lambda kv: 'backbone' not in kv[0], model.named_parameters()))
        backbone_param = list(map(lambda x: x[1], backbone_param))
        other_param = list(map(lambda x: x[1], other_param))
        optimizer = torch.optim.AdamW([{'params': backbone_param, 'lr': config.train.lr},
                                       {'params': other_param, 'lr': config.train.lr}],
                                       lr=config.train.lr,
                                       weight_decay=config.train.weight_decay)
    return optimizer


def get_schedular(config, optimizer):
    milestones = config.train.schedualr_milestones
    gamma = config.train.schedular_gamma
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    return schedular


def set_model_train(config, model, ddp):        # TODO: fix other parameters
    model.train()
    if config.model.fix_backbone:
        if ddp:
            model.module.backbone.eval()
        else:
            model.backbone.eval()

def load_pretrain(model, pretrain_cpt_name, strict=False, device=None):
    if os.path.isfile(pretrain_cpt_name):
        print(f"=> loading pretrain from {pretrain_cpt_name}")
        if device is not None:
            checkpoint = torch.load(pretrain_cpt_name, map_location=device)
        else:
            checkpoint = torch.load(pretrain_cpt_name, map_location=torch.device('cpu'))
        
        # load model
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]

        # missing_states = missing_states - set(state_dict.keys())
        # if len(missing_states) > 0:
        #     warnings.warn("Loading Pretrain State:\nMissing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(pretrain_cpt_name))

def resume_training(model, optimizer, schedular, scaler, output_dir, cpt_name='', strict=True, device=None):
    output_dir = os.path.join(output_dir, cpt_name)
    if os.path.isfile(output_dir):
        print("=> loading checkpoint {}".format(output_dir))
        if device is not None:
            checkpoint = torch.load(output_dir, map_location=device)
        else:
            checkpoint = torch.load(output_dir, map_location=torch.device('cpu'))
        
        # load model
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        # missing_states = missing_states - set(state_dict.keys())
        # if len(missing_states) > 0:
        #     warnings.warn("Resume Training State\nMissing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)

        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])

        # load schedular
        schedular.load_state_dict(checkpoint['schedular'])

        # load scaler
        scaler.load_state_dict(checkpoint['scaler'])

        # load epoch
        start_epoch = checkpoint['epoch']

        # load data
        best_iou = checkpoint['best_iou'] if 'best_iou' in checkpoint.keys() else 0.0
        best_prob = checkpoint['best_prob'] if 'best_prob' in checkpoint.keys() else float('-inf')

        return model, optimizer, schedular, scaler, start_epoch, best_iou, best_prob
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(output_dir))


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def store_trainable(model, config):
    # for name, para in model.named_parameters():
    #     if para.requires_grad == True:
    #         trainable_list.append(name)
    
    trainable_list = set([n for n,p in model.named_parameters() if p.requires_grad])
    save_set = set()
    for key in model.state_dict().keys():
        module = key.split('.')[0]
        if module in config.train.trainable_module:
            save_set.add(key)
    save_set = save_set | trainable_list
    return {k:v for k, v in model.state_dict().items() if k in save_set}