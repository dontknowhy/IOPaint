import torch

from ..utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet


def init_parsing_model(model_name='bisenet', half=False, device='cuda', model_rootpath=None):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
        model_hash = '468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
        model_hash = '3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath,
        hash_prefix=model_hash)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
