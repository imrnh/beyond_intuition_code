import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

_logger = logging.getLogger(__name__)

"""
     Load pretrained model.
     
     For dino, the url_linear contain weight an bias for the head that needed to be injected into backbone. 
     Cause, the backbone doesn't have a classifier. Instead, is a very good feature extractor.
     
     No other model type require this processing.
     
"""


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, mae=False, moco=False, dino=False):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')

    if dino:
        state_backbone = torch.load(cfg['url_backbone'], map_location='cpu')
        state_linear = torch.load(cfg['url_linear'], map_location='cpu')[
            'state_dict']  # Get weight of the last layer only.
        state_dict = state_backbone.copy()
        state_dict['head.weight'] = state_linear[
            'module.linear.weight']  # Shape: (1000, 1536). That means, the final layer would have input from previous fc layer with 1536 neurons.
        state_dict['head.bias'] = state_linear['module.linear.bias']  # Shape: (1000,)
    else:
        state_dict = torch.load(cfg['url'], map_location='cpu')

    if mae:
        state_dict = state_dict['model']
    if moco:
        state_dict = state_dict['state_dict']
        for i in list(state_dict.keys()):
            name = i.split('module.')[1]
            state_dict[name] = state_dict.pop(i)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure its float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)  # For models with space2depth stems
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:  # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[
                                                  1:]  # Class index 0 give to background. Therefore, starting from index 1.
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    model.load_state_dict(state_dict, strict=strict)


"""
    Load state dictionary of the pretrained model. 

"""


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
