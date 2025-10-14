def build_TransMorph(img_size):
    from networks.transmorph.TransMorph import TransMorph
    from networks.transmorph.TransMorph import CONFIGS as CONFIGS_TM
    config = CONFIGS_TM['TransMorph']
    return TransMorph(config, img_size)

def build_CorrMLP(img_size):
    from networks.corrmlp.CorrMLP import CorrMLP
    return CorrMLP()

def build_GroupMorph(img_size):
    from networks.groupmorph.GroupMorph import GroupMorph
    groups = (4, 2, 2)  # (4,4,4), (4,4,2), (4,2,2) or (2,2,2)
    return GroupMorph(1, 8, img_size, groups)

def build_IIRPNet(img_size):
    from networks.iirpnet.iirp import RPNet
    return RPNet(img_size)

def build_CGNet(img_size):
    from networks.cgnet.CGNet import CGNet
    return CGNet()

def build_EfficientMorph(img_size):
    from networks.efficientmorph.EfficientMorph import EfficientMorph
    from networks.efficientmorph.EfficientMorph import CONFIGS as CONFIGS_TM
    config = CONFIGS_TM['EfficientMorph_2x3_2']
    return EfficientMorph(config, img_size)

MODEL_FACTORY = {
    "TransMorph": build_TransMorph,
    "CorrMLP": build_CorrMLP,
    "GroupMorph": build_GroupMorph,
    "IIRPNet": build_IIRPNet,
    "CGNet": build_CGNet,
    "EfficientMorph": build_EfficientMorph,
}

def build_model(model_label, img_size):
    if model_label not in MODEL_FACTORY:
        raise ValueError(f"Unknown model label: {model_label}")
    return MODEL_FACTORY[model_label](img_size)
