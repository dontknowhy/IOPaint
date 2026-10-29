import torch
from copy import deepcopy

from ..utils import load_file_from_url
from .retinaface import RetinaFace


def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
        model_hash = '6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
        model_hash = '2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath,
        hash_prefix=model_hash)

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
