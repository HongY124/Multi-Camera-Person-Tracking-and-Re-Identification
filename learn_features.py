
import os
import cv2
import numpy as np
import argparse
import torch
import sys
import torch.nn.functional as F
from ultralytics import YOLO

# Import Encoder
try:
    from torchreid_encoder import create_box_encoder
except ImportError:
    print("Cannot find torchreid_encoder.py")
    sys.exit(1)

from feature_manager import FeatureDatabase

def main():
    parser = argparse.ArgumentParser(description="Feature Learning (AIN)")
    parser.add_argument('--targets', default='targets', help='Directory with person images')
    # Use AIN (Adaptive Instance Norm) for better domain generalization
    parser.add_argument('--reid_model', default='osnet_ain_x1_0', help='ReID Model')
    parser.add_argument('--force', action='store_true', help='Clear DB')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== GENERALIZED FEATURE LEARNING (OSNet-AIN) ===")
    
    # 1. Models
    print("Loading Detection...")
    try: seg_model = YOLO('yolov8m-seg.pt')
    except: seg_model = YOLO('yolov8m.pt')
    
    print(f"Loading ReID: {args.reid_model}...")
    encoder = create_box_encoder(model_name=args.reid_model, device=device)
    
    # 2. Database
    db = FeatureDatabase(db_folder="features_db")
    if args.force:
        print("Clearing database...")
        db.data = {}
        
    # 3. Data Loading
    files = []
    if os.path.exists(args.targets):
        for root, dirs, filenames in os.walk(args.targets):
            for f in filenames:
                if f.lower().endswith(('.jpg','.png','.jpeg')):
                    files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} target images.")
    
    for img_path in files:
        # Infer name
        parent = os.path.basename(os.path.dirname(img_path))
        if parent == 'targets':
            name = os.path.splitext(os.path.basename(img_path))[0]
        else:
            name = parent
            
        print(f"Processing: {name} | {os.path.basename(img_path)}")
        
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # A. Detect Person & Masking
        results = seg_model(frame, classes=0, verbose=False)
        clean_crop = None
        
        if len(results[0].boxes) > 0:
            # Find largest person
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in results[0].boxes.xyxy.cpu().numpy()]
            idx = np.argmax(areas)
            
            # Masking
            if results[0].masks is not None:
                mask = results[0].masks.data[idx].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                binary_mask = (mask > 0.5).astype(np.uint8)
                masked_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
            else:
                masked_frame = frame
                
            box = results[0].boxes.xyxy.cpu().numpy()[idx]
            x1,y1,x2,y2 = map(int, box)
            clean_crop = masked_frame[y1:y2, x1:x2]
        else:
            # Fallback: Use whole image if cropped manually
            clean_crop = frame

        if clean_crop is None or clean_crop.size == 0: continue
        
        # B. TEST TIME AUGMENTATION (TTA)
        # We need variety in the gallery to match the variety in the video
        aug_crops = []
        aug_names = []
        
        # 1. Base
        aug_crops.append(clean_crop); aug_names.append("orig")
        
        # 2. Flip
        aug_crops.append(cv2.flip(clean_crop, 1)); aug_names.append("flip")
        
        # 3. Zoom
        h, w = clean_crop.shape[:2]
        pad_h, pad_w = int(h*0.1), int(w*0.1)
        zoom_crop = clean_crop[pad_h:h-pad_h, pad_w:w-pad_w]
        if zoom_crop.size > 0:
            aug_crops.append(zoom_crop); aug_names.append("zoom")

        # C. Embedding Extraction (CENTROID LEARNING)
        try:
            feats = encoder(aug_crops, return_tensors=True) # (N, 512)
            if hasattr(feats, 'float'): feats = feats.float()
            
            # --- CORE LOGIC: FEATURE AGGREGATION ---
            # Thay vì luu tung bien the (de bi hoc vet anh), ta tinh TRUNG BINH CONG
            # De tao ra 1 vector 'Tinh Hoa' (Centroid) dai dien cho nguoi do
            
            centroid_feat = torch.mean(feats, dim=0, keepdim=True) # (1, 512)
            centroid_feat = F.normalize(centroid_feat, p=2, dim=1).cpu().numpy() # L2 Norm
            
            # Save ONLY the Centroid
            key = f"{name}_{os.path.basename(img_path)}_robust"
            db.add_feature(name, key, centroid_feat[0], meta_info={'path': img_path})
            
        except Exception as e:
            print(f"Error encoding: {e}")
            continue
            
    db.save()
    print("=== DONE. ReID Database Updated (AIN). ===")

if __name__ == "__main__":
    main()

