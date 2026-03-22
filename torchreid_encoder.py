import torch
import torchreid
from torchreid.utils import FeatureExtractor
import numpy as np
import cv2

class TorchReIDEncoder:
    def __init__(self, model_name='osnet_x1_0', device='cuda', model_path=None):
        self.device = device
        print(f"[Info] Loading ReID Model: {model_name}...")
        
        # FeatureExtractor tự động tải model về nếu chưa có
        # Nếu ní đã có file .pth offline, có thể chỉ định model_path (tùy implementation của thư viện)
        # Nhưng mặc định torchreid sẽ tự download vào ~/.cache/torch/checkpoints/
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=device,
            image_size=(256, 128)
        )

    def __call__(self, input_data, boxes=None, return_tensors=False):
        # input_data: Có thể là Ảnh gốc BGR (numpy) HOẶC List các ảnh crop sẵn (list of numpy)
        # boxes: List [x, y, w, h] (Chỉ dùng nếu input_data là ảnh gốc)
        
        patches = []
        
        # TRƯỜNG HỢP 1: Input là List ảnh crop sẵn (Pre-cropped / Masked)
        if isinstance(input_data, list):
            for patch in input_data:
                # Convert BGR -> RGB nếu cần (Giả sử input cũng là BGR từ OpenCV)
                if patch is not None and patch.size > 0:
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    patches.append(patch_rgb)
                else:
                    patches.append(np.zeros((256, 128, 3), dtype=np.uint8))
        
        # TRƯỜNG HỢP 2: Input là 1 ảnh gốc -> Cần cắt theo boxes
        elif isinstance(input_data, np.ndarray) and boxes is not None:
            ori_img = input_data
            for box in boxes:
                x, y, w, h = map(int, box)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(ori_img.shape[1], x + w)
                y2 = min(ori_img.shape[0], y + h)
                
                if x2 <= x1 or y2 <= y1:
                    patch = np.zeros((256, 128, 3), dtype=np.uint8)
                else:
                    patch = ori_img[y1:y2, x1:x2]
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                
                patches.append(patch)
        else:
            print("[Error] Encoder input không hợp lệ.")
            empty = torch.empty(0) if return_tensors else np.array([])
            return empty

        if not patches:
            empty = torch.empty(0) if return_tensors else np.array([])
            return empty

        # Trích xuất feature
        features = self.extractor(patches) # Tensor on device
        
        if return_tensors:
            return features # Return tensor directly (on GPU if model is on GPU)
            
        return features.cpu().numpy()

def create_box_encoder(model_name='osnet_x1_0', device='cuda'):
    return TorchReIDEncoder(model_name, device)