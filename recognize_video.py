
import os
import cv2
import pickle
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

def draw_text(img, text, pos, color=(0, 255, 0), scale=0.6):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (x, y - h - 5), (x + w, y), color, -1)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--reid_model', default='osnet_ain_x1_0')
    parser.add_argument('--threshold', type=float, default=0.68) # Strict threshold to reject 0.66 FPs
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== DEEP REID INFERENCE (AIN - STRICT MODE) ===")
    
    # 1. Load DB
    db = FeatureDatabase(db_folder="features_db")
    flat_data = db.get_all_features()
    if not flat_data:
        print("Database empty. Run learn_features.py --force")
        return
        
    gallery_vecs = []
    gallery_names = []
    
    for item in flat_data:
        v = item['feat']
        if isinstance(v, np.ndarray): v = torch.from_numpy(v)
        v = v.float().to(device)
        
        # Ensure 512-dim
        if v.shape[0] != 512: continue
           
        gallery_vecs.append(v.unsqueeze(0))
        gallery_names.append(item['name'])
        
    if not gallery_vecs:
        print("No valid 512-dim features found. Please RE-LEARN.")
        return
        
    gallery_tensor = torch.cat(gallery_vecs, dim=0) # (N, 512)
    gallery_tensor = F.normalize(gallery_tensor, p=2, dim=1)
    print(f"Loaded {len(gallery_names)} identities.")
    
    # 2. Models
    print("Loading YOLO...")
    try: yolo = YOLO('yolov8m-seg.pt')
    except: yolo = YOLO('yolov8m.pt')
    
    print(f"Loading ReID {args.reid_model}...")
    encoder = create_box_encoder(model_name=args.reid_model, device=device)
    
    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(3)); h = int(cap.get(4))
    fps = cap.get(5)
    out_path = os.path.join("videos", "output", f"final_{os.path.basename(args.video)}")
    if not os.path.exists("videos/output"): os.makedirs("videos/output")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    tracker_state = {}
    ENTRY_THRESH = args.threshold # 0.72
    KEEP_THRESH = 0.65 # Require consistently high score to maintain lock
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        candidates = {}
        
        # Track
        results = yolo.track(frame, persist=True, tracker='custom_tracker.yaml', verbose=False, classes=0, conf=0.5)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            
            crops = []
            valid_indices = []
            min_h = h // 10
            
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = map(int, box)
                x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                
                # Filters
                if (y2-y1) < min_h: continue # Too small
                
                ar = (y2-y1)/(x2-x1) if (x2-x1)>0 else 0
                if ar < 1.0 or ar > 4.5: continue
                
                # Mask
                if results[0].masks is not None:
                    m = results[0].masks.data[i].cpu().numpy()
                    m = cv2.resize(m, (w, h))
                    m_crop = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8)
                    person = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=m_crop)
                else:
                    person = frame[y1:y2, x1:x2]
                    
                crops.append(person)
                valid_indices.append(i)
                
            if crops:
                # 1. TTA Probe (Orig + Flip)
                crops_flip = [cv2.flip(c, 1) for c in crops]
                
                feats_orig = encoder(crops, return_tensors=True)
                feats_flip = encoder(crops_flip, return_tensors=True)
                
                if hasattr(feats_orig, 'float'): 
                    feats_orig = feats_orig.float()
                    feats_flip = feats_flip.float()
                    
                # Normalize
                feats_orig = F.normalize(feats_orig, p=2, dim=1)
                feats_flip = F.normalize(feats_flip, p=2, dim=1)
                
                # Average (Fusion)
                scene_feats = (feats_orig + feats_flip) / 2.0
                scene_feats = F.normalize(scene_feats, p=2, dim=1) # Renormalize
                
                # 2. MATCHING (Cosine Sim)
                sim_mat = torch.mm(scene_feats, gallery_tensor.t()) # (Batch, N_Gallery)
                
                # 3. ROBUST SCORING (Top-K Voting)
                # Chuyen doi sim_mat sang CPU list de xu ly logic phuc tap
                all_scores = sim_mat.cpu().tolist()
                
                for k, orig_idx in enumerate(valid_indices):
                    curr_id = ids[orig_idx]
                    
                    # Logic Score cho tung Person ID trong Gallery
                    # Thay vi lay Max, ta gom nhom theo Ten
                    person_scores = {}
                    for g_idx, score in enumerate(all_scores[k]):
                        name = gallery_names[g_idx]
                        if name not in person_scores: person_scores[name] = []
                        person_scores[name].append(score)
                        
                    # Tinh diem dai dien cho moi Person Name (Average Top-3)
                    best_name = 'Stranger'
                    best_avg_score = 0.0
                    
                    for name, s_list in person_scores.items():
                        s_list.sort(reverse=True)
                        # Take average of Top-3 (or fewer if not enough)
                        k_top = min(len(s_list), 3)
                        avg_score = sum(s_list[:k_top]) / k_top
                        
                        if avg_score > best_avg_score:
                            best_avg_score = avg_score
                            best_name = name
                            
                    raw_score = best_avg_score
                    name = best_name
                    
                    if curr_id not in tracker_state:
                         tracker_state[curr_id] = {'name': 'Stranger', 'conf': 0, 'locked': False}
                    
                    inf = tracker_state[curr_id]
                    
                    # State Machine logic
                    ENTRY_THRESH = 0.73 # Raised to reject 0.71 False Positives
                    KEEP_THRESH = 0.63 
                    
                    if inf['locked']:
                        # Maintain lock if score is decent
                        if raw_score > KEEP_THRESH:
                             # Update name if it's a stronger match for a different person (unlikely but possible)
                             if name != inf['name'] and raw_score > ENTRY_THRESH:
                                  inf['name'] = name
                             # Keep locked
                        else:
                             # Dropped below Keep Threshold -> Reset
                             inf['locked'] = False
                             inf['name'] = 'Stranger'
                             inf['conf'] = 0
                    else:
                        # Search
                        if raw_score > ENTRY_THRESH:
                             inf['conf'] += 1
                             if inf['conf'] >= 5: # Require 5 frames (Strict Stability)
                                 inf['locked'] = True
                                 inf['name'] = name
                        else:
                             inf['conf'] = max(0, inf['conf'] - 1)
                             
                    # Stage 1: Candidate Gathering
                    if inf['locked'] and inf['name'] != 'Stranger':
                        if inf['name'] not in candidates: candidates[inf['name']] = []
                        candidates[inf['name']].append({
                            'score': raw_score,
                            'box': boxes[orig_idx],
                            'tracker_id': curr_id
                        })
                        
        # Stage 2: Unique Draw
        for name, matches in candidates.items():
            matches.sort(key=lambda x: x['score'], reverse=True)
            winner = matches[0]
            
            x1,y1,x2,y2 = map(int, winner['box'])
            label = f"{name} ({winner['score']:.2f})"
            color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_text(frame, label, (x1, y1), color)
            
        out.write(frame)
        if frame_idx % 50 == 0: print(f"Frame {frame_idx}...")
        
    cap.release(); out.release()
    print(f"Done: {out_path}")

if __name__ == "__main__":
    main()
