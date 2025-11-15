import numpy as np
import cv2
from scipy.ndimage import convolve, zoom
from PIL import Image



def pad_to_multiple(image: np.ndarray, multiple: int = 8):
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, h, w

def crop_to_original(image: np.ndarray, h: int, w: int):
    return image[:h, :w]

def wavelet_blur_np(image: np.ndarray, radius: int):
    kernel = np.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625]
    ], dtype=np.float32)

    blurred = np.empty_like(image)
    for c in range(image.shape[0]):
        blurred_c = convolve(image[c], kernel, mode='nearest')
        if radius > 1:
            blurred_c = zoom(zoom(blurred_c, 1 / radius, order=1), radius, order=1)
        blurred[c] = blurred_c
    return blurred

def wavelet_decomposition_np(image: np.ndarray, levels=5):
    high_freq = np.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur_np(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq

def wavelet_reconstruction_np(content_feat: np.ndarray, style_feat: np.ndarray):
    content_high, _ = wavelet_decomposition_np(content_feat)
    _, style_low = wavelet_decomposition_np(style_feat)
    return content_high + style_low

def wavelet_color_fix_np(fused: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fused_np = fused.astype(np.float32) / 255.0
    mask_np = mask.astype(np.float32) / 255.0

    fused_np = fused_np.transpose(2, 0, 1)
    mask_np = mask_np.transpose(2, 0, 1)

    result_np = wavelet_reconstruction_np(fused_np, mask_np)

    result_np = result_np.transpose(1, 2, 0)
    result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

    return result_np

def attention_guided_fusion(ori: np.ndarray, removed: np.ndarray, attn_map: np.ndarray, multiple: int = 8):
    H, W = ori.shape[:2]
    attn_map = attn_map.astype(np.float32)
    _, attn_map = cv2.threshold(attn_map, 128, 255, cv2.THRESH_BINARY)
    am = attn_map.astype(np.float32)
    am = am/255.0
    am_up = cv2.resize(am, (W, H), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    am_d = cv2.dilate(am_up, kernel, iterations=1)
    am_d = cv2.GaussianBlur(am_d.astype(np.float32), (9,9), sigmaX=2)

    am_merged = np.maximum(am_up, am_d)
    am_merged = np.clip(am_merged, 0, 1)

    attn_up_3c = np.stack([am_merged]*3, axis=-1)
    attn_up_ori_3c = np.stack([am_up]*3, axis=-1)

    ori_out = ori * (1 - attn_up_ori_3c)
    rem_out = removed * (1 - attn_up_ori_3c)

    ori_pad, h0, w0 = pad_to_multiple(ori_out, multiple)
    rem_pad, _, _   = pad_to_multiple(rem_out, multiple)

    wave_rgb = wavelet_color_fix_np(ori_pad, rem_pad)
    wave = crop_to_original(wave_rgb, h0, w0)
    # fusion
    fused = (wave * (1 - attn_up_3c) + removed * attn_up_3c).astype(np.uint8)
    return fused


def resize_by_short_side(image, target_short=512, resample=Image.BICUBIC):
    w, h = image.size
    if w < h:
        new_w = target_short
        new_h = int(h * target_short / w)
        new_h = (new_h + 15) // 16 * 16 
    else:
        new_h = target_short
        new_w = int(w * target_short / h)
        new_w = (new_w + 15) // 16 * 16
    return image.resize((new_w, new_h), resample=resample)