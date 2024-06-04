import torch
import cv2
import kornia as K


##image loading and resize
def load_torch_image(fname, device=torch.device("cpu")):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.0
    img = K.color.bgr_to_rgb(img.to(device))
    return img


def resize_torch_image(
    timg, resize_long_edge_to=None, align=None, disable_enlarge=True
):
    h, w = timg.shape[2:]
    raw_size = torch.tensor(timg.shape[2:])
    if resize_long_edge_to is None:
        scale = 1
    else:
        scale = float(resize_long_edge_to) / float(max(raw_size[0], raw_size[1]))

    if disable_enlarge:
        scale = min(scale, 1)

    h_resized = int(h * scale)
    w_resized = int(w * scale)

    if align is not None:
        assert align > 0
        h_resized = h_resized - h_resized % align
        w_resized = w_resized - w_resized % align
    scale_h = h_resized / h
    scale_w = w_resized / w

    timg_resized = K.geometry.resize(timg, (h_resized, w_resized), antialias = True)
    return timg_resized, scale_h, scale_w


def get_roi_image(timg, roi):
    min_h = int(roi["roi_min_h"])
    min_w = int(roi["roi_min_w"])
    max_h = int(roi["roi_max_h"])
    max_w = int(roi["roi_max_w"])
    roi_img = timg[:, :, min_h:max_h, min_w:max_w]
    roi_w_scale = (max_w - min_w) / timg.shape[3]
    roi_h_scale = (max_h - min_h) / timg.shape[2]
    return roi_img, min_h, min_w