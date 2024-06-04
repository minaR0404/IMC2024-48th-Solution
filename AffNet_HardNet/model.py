import os
import math
import torch
import kornia.feature as KF

from config.config import MODEL_DIR, HARDNET_PT


##affnethardnet model
# Making kornia local features loading w/o internet
class AffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
        self,
        num_features: int = 5000,
        upright: bool = False,
        device=torch.device("cpu"),
        scale_laf: float = 1.0,
        detector = "keynet"
    ):
        detector_options = ["keynet", "GFTT", "Hessian", "Harris", "DoG"]
        if detector not in detector_options:
            raise ValueError("Detector must be one of {}".format(detector_options))
        
        ori_module = (
            KF.PassLAF()
            if upright
            else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        )
        if not upright:
            weights = torch.load(os.path.join(MODEL_DIR, "OriNet.pth"))["state_dict"]
            ori_module.angle_detector.load_state_dict(weights)

        config = {
            # Extraction Parameters
            "nms_size": 15,
            "pyramid_levels": 4,
            "up_levels": 1,
            "scale_factor_levels": math.sqrt(2),
            "s_mult": 22.0,
        }

        if detector == "keynet":
            detector = KF.KeyNetDetector(
            False,
            num_features=num_features,
            ori_module=ori_module,
            aff_module=KF.LAFAffNetShapeEstimator(False).eval(),
            ).to(device)
            kn_weights = torch.load(os.path.join(MODEL_DIR, "keynet_pytorch.pth"))[
            "state_dict"
            ]
            detector.model.load_state_dict(kn_weights)
        elif detector == "GFTT":
            detector = KF.MultiResolutionDetector(
                KF.CornerGFTT(),
                num_features=num_features,
                config=config,
                ori_module=ori_module,
                aff_module=KF.LAFAffNetShapeEstimator(False).eval(),
            ).to(device)
        elif detector == "Harris":
            detector = KF.MultiResolutionDetector(
                KF.CornerHarris(0.04),
                num_features=num_features,
                config=config,
                ori_module=ori_module,
                aff_module=KF.LAFAffNetShapeEstimator(False).eval(),
            ).to(device)
        elif detector == "DoG":
            detector = KF.MultiResolutionDetector(
                KF.BlobDoGSingle(),
                num_features=num_features,
                config=config,
                ori_module=ori_module,
                aff_module=KF.LAFAffNetShapeEstimator(False).eval(),
            ).to(device)
        affnet_weights = torch.load(os.path.join(MODEL_DIR, "AffNet.pth"))["state_dict"]
        detector.aff.load_state_dict(affnet_weights)

#         hardnet = KF.HardNet(False).eval()
#         hn_weights = torch.load(os.path.join(MODEL_DIR, "HardNetLib.pth"))["state_dict"]
#         hardnet.load_state_dict(hn_weights)
#         descriptor2 = KF.LAFDescriptor(
#             hardnet, patch_size=32, grayscale_descriptor=True
#         ).to(device)
        hardnet8 = KF.HardNet8(False).eval()
        hn8_weights = torch.load(HARDNET_PT)
        hardnet8.load_state_dict(hn8_weights)
        descriptor = KF.LAFDescriptor(
            hardnet8, patch_size=32, grayscale_descriptor=True
        ).to(device)
        super().__init__(detector, descriptor, scale_laf)