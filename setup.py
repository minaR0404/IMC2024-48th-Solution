import torch
from config import MODEL_DICT, device
from AffNet_HardNet.model import AffNetHardNet
from AffNet_HardNet.detector import AffNetHardNetDetector
from AffNet_HardNet.matcher import LafMatcher
from DeDoDe_main.DeDoDe import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
from DeDoDeV2.detector import DeDoDeDetectorV2
from DeDoDeV2.matcher import DualSoftMaxMatcherV2, DeDoDeMatcherV2


##model setup
if MODEL_DICT["Keynet"]["enable"]:
    keynet_model = (
        AffNetHardNet(num_features=8000, upright=False, device=device, detector="keynet")
        .to(device)
        .eval()
    )
    keynet_detector = AffNetHardNetDetector(keynet_model, resize_long_edge_to=MODEL_DICT["Keynet"]["resize_long_edge_to"])
    laf_matcher = LafMatcher(device=device)
    
if MODEL_DICT["GFTT"]["enable"]:
    gftt_model = (
        AffNetHardNet(num_features=8000, upright=False, device=device, detector="GFTT")
        .to(device)
        .eval()
    )
    gftt_detector = AffNetHardNetDetector(gftt_model, resize_long_edge_to=MODEL_DICT["GFTT"]["resize_long_edge_to"])
    laf_matcher = LafMatcher(device=device)

if MODEL_DICT["DoG"]["enable"]:
    DoG_model = (
        AffNetHardNet(num_features=8000, upright=False, device=device, detector="DoG")
        .to(device)
        .eval()
    )
    DoG_detector = AffNetHardNetDetector(DoG_model, resize_long_edge_to=MODEL_DICT["DoG"]["resize_long_edge_to"])
    laf_matcher = LafMatcher(device=device)

if MODEL_DICT["Harris"]["enable"]:
    harris_model = (
        AffNetHardNet(num_features=8000, upright=False, device=device, detector="Harris")
        .to(device)
        .eval()
    )
    harris_detector = AffNetHardNetDetector(harris_model, resize_long_edge_to=MODEL_DICT["Harris"]["resize_long_edge_to"])
    laf_matcher = LafMatcher(device=device)
    
if MODEL_DICT["DeDoDe"]["enable"]:
    dedode_detector = DeDoDeDetectorV2(detector = dedode_detector_L(weights=torch.load("/kaggle/input/dedode/dedode_detector_L_v2.pth"), device=device),
                                     descriptor = dedode_descriptor_G(weights=torch.load("/kaggle/input/dedode/dedode_descriptor_G.pth"), 
                                                                      dinov2_weights=torch.load("/kaggle/input/dinov2-vit-pretrain/dinov2_vitl14_pretrain.pth"), 
                                                                      device=device), 
                                     device=device)
    dedode_matcher = DeDoDeMatcherV2(matcher = DualSoftMaxMatcherV2(),
                                     device=device)