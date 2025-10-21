import torch



if __name__ == '__main__':
    checkpoint_path = '/18018998051/SAMWISE/pretrain/CLAP/630k-best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']