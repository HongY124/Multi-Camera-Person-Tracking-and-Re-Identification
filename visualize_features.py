
import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor

def visualize_directory(input_path, model_name='osnet_ain_x1_0'):
    """
    Minh họa quá trình học sâu (Deep Learning Visualization)
    """
    # 1. KHỞI TẠO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"==================================================")
    print(f"   CÔNG CỤ PHÂN TÍCH TRI THỨC AI (VISUALIZER)   ")
    print(f"   Thiết bị: {device.upper()}")
    print(f"==================================================")

    # Model 1: Tách nền (YOLO Segmentation)
    print("[Step 1] Loading Preprocessing Model (YOLOv8-Seg)...")
    yolo_seg = YOLO('yolov8m-seg.pt')

    # Model 2: Deep Ranking (ReID)
    print(f"[Step 2] Loading Deep Learning Model ({model_name})...")
    extractor = FeatureExtractor(
        model_name=model_name,
        device=device,
        image_size=(256, 128)
    )
    model = extractor.model
    model.eval()

    # 2. GẮN HOOK VÀO CÁC TẦNG CỦA MẠNG NEURAL
    # Chúng ta muốn xem cái gì đang diễn ra bên trong "hộp đen"
    # Mục tiêu: Lấy đại diện Tầng Nông -> Tầng Giữa -> Tầng Sâu

    feature_maps = {} 
    def get_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple): feature_maps[name] = output[0]
            else: feature_maps[name] = output
        return hook_fn

    # Tìm các layer conv/sequential
    all_layers = list(model.named_children())
    valid_layers = [(n, m) for n, m in all_layers if len(list(m.parameters())) > 0]
    
    # Chọn 3 đại diện tiêu biểu (Đầu, Giữa, Cuối)
    if len(valid_layers) >= 3:
        idxs = [0, len(valid_layers)//2, len(valid_layers)-1]
        selected = [valid_layers[i] for i in idxs]
        labels = ["Tang 1: HOC CANH/GOC", "Tang 2: HOC HINH DANG", "Tang 3: HOC DOI TUONG"]
    else:
        selected = valid_layers
        labels = [f"Layer {i}" for i in range(len(selected))]

    hooks = []
    print(f"[Info] Đã kết nối vào {len(selected)} điểm kiểm tra:")
    for i, (name, module) in enumerate(selected):
        module.register_forward_hook(get_hook(name))
        print(f"  + {labels[i]} (Tại module: {name})")

    # 3. XỬ LÝ ẢNH
    image_files = []
    if os.path.isfile(input_path): image_files.append(input_path)
    else:
        for root, dirs, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(('.jpg','.png','.jpeg')):
                    image_files.append(os.path.join(root, f))
    
    print(f"[Process] Tìm thấy {len(image_files)} ảnh cần phân tích.")

    for i, img_path in enumerate(image_files):
        print(f"  > Đang giải mã não bộ AI cho ảnh: {os.path.basename(img_path)}")
        feature_maps.clear()
        
        # A. Preprocessing (Tách nền)
        img = cv2.imread(img_path)
        if img is None: continue
        
        seg_res = yolo_seg(img, verbose=False, classes=0)
        img_clean = img.copy()
        
        if len(seg_res[0].boxes) > 0:
            # Lấy mask
            best_idx = np.argmax(seg_res[0].boxes.conf.cpu().numpy())
            box = seg_res[0].boxes.xyxy[best_idx].cpu().numpy()
            
            if seg_res[0].masks:
                mask = seg_res[0].masks.data[best_idx].cpu().numpy()
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                binary_mask = (mask > 0.5).astype(np.uint8)
                img_clean = cv2.bitwise_and(img, img, mask=binary_mask)
            
            x1,y1,x2,y2 = map(int, box)
            x1=max(0,x1); y1=max(0,y1)
            crop = img_clean[y1:y2, x1:x2]
        else:
            crop = img
            
        if crop.size == 0: continue
            
        # B. Deep Learning Pass
        # Resize to network input
        ih, iw = 384, 192 # Độ phân giải cao
        inp = cv2.resize(crop, (iw, ih))
        
        # Normalize
        tensor = inp[:,:,::-1].astype(np.float32) / 255.0
        tensor = (tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(tensor.transpose(2,0,1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(tensor) # Forward pass

        # C. Visualization (Vẽ Heatmap)
        viz_row = []
        
        # 1. Ảnh gốc sạch (Clean Input)
        ref_img = cv2.resize(crop, (150, 300))
        cv2.putText(ref_img, "INPUT THO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        viz_row.append(ref_img)

        # 2. Các tầng đặc trưng
        for j, (name, _) in enumerate(selected):
            if name not in feature_maps: continue
            
            # Lấy activation map
            fmap = feature_maps[name] # [1, C, H, W]
            
            # Tính trung bình các kênh (Average Channel Activation)
            heatmap = torch.mean(fmap, dim=1).squeeze().cpu().numpy()
            
            # Normalize 0-255
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) + 1e-8
            heatmap = np.uint8(255 * heatmap)
            
            # Colorize
            heatmap_img = cv2.resize(heatmap, (150, 300))
            heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_TURBO) # Turbo đẹp hơn Jet
            
            # Label
            cv2.putText(heatmap_color, labels[j], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            viz_row.append(heatmap_color)

        # Ghép ảnh
        final_img = np.hstack(viz_row)
        
        # Lưu
        fname = os.path.basename(img_path)
        out_folder = f"analysis_results/{os.path.basename(os.path.dirname(img_path))}"
        if not os.path.exists(out_folder): os.makedirs(out_folder)
        
        cv2.imwrite(os.path.join(out_folder, f"DeepViz_{fname}"), final_img)

    print(f"\n[XONG] Kết quả đã lưu tại thư mục: analysis_results/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='targets', help="Folder ảnh")
    args = parser.parse_args()
    visualize_directory(args.input)
