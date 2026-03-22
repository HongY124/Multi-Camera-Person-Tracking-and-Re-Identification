
import os
import pickle

class FeatureDatabase:
    def __init__(self, db_folder="features_db", db_name="known_features.pkl"):
        # Đảm bảo thư mục tồn tại
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
            print(f"[DB] Đã tạo thư mục lưu trữ: {db_folder}")
            
        self.db_path = os.path.join(db_folder, db_name)
        self.data = self._load()

    def _load(self):
        """Tải DB từ file lên bộ nhớ"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    print(f"[DB] Đang tải dữ liệu từ {self.db_path}...")
                    return pickle.load(f)
            except Exception as e:
                print(f"[DB] Lỗi đọc file {self.db_path}: {e}. Khởi tạo DB mới.")
        return {}

    def save(self):
        """Lưu bộ nhớ xuống file (dạng binary và dạng readable json)"""
        # 1. Lưu Binary (cho máy đọc)
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.data, f)
        
        # 2. Lưu Text/JSON (cho người đọc)
        import json
        readable_data = {}
        for name, imgs in self.data.items():
            readable_data[name] = []
            for img_key, info in imgs.items():
                # Chỉ lưu metadata và path, không lưu vector dài ngoằng
                # Convert numpy types to native python types for JSON serialization
                crop_h = info.get("meta", {}).get("crop_h", 0)
                if hasattr(crop_h, 'item'): crop_h = crop_h.item() # Numpy scalar -> Python scalar
                
                feat_sample = info["feat"][:5]
                if hasattr(feat_sample, 'tolist'): feat_sample = feat_sample.tolist()

                readable_entry = {
                    "source_image": info.get("meta", {}).get("path", "Unknown"),
                    "image_key": img_key,
                    "resolution_height": int(crop_h),
                    "feature_vector_sample": str(feat_sample) + " ...", 
                    "status": "Learned"
                }
                readable_data[name].append(readable_entry)
        
        manifest_path = self.db_path.replace('.pkl', '.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(readable_data, f, indent=4, ensure_ascii=False)

        print(f"[DB] Đã lưu dữ liệu vào: \n   + {self.db_path} (Máy)\n   + {manifest_path} (Người đọc)")

    def add_feature(self, person_name, img_key, feature_vector, meta_info=None):
        """Thêm đặc trưng mới"""
        if person_name not in self.data:
            self.data[person_name] = {}
        
        self.data[person_name][img_key] = {
            "feat": feature_vector,
            "meta": meta_info or {}
        }

    def has_image(self, person_name, img_key):
        """Kiểm tra ảnh này đã học chưa"""
        if person_name in self.data:
            if img_key in self.data[person_name]:
                return True
        return False

    def get_all_features(self):
        """Trả về list phẳng để dùng cho vòng lặp so sánh nhanh"""
        flat_list = []
        for name, images in self.data.items():
            for key, item in images.items():
                flat_list.append({
                    "key": key,
                    "name": name,
                    "feat": item["feat"],
                    "meta": item.get("meta", {})
                })
        return flat_list

    def get_summary(self):
        """Trả về thống kê"""
        summary = {}
        total_samples = 0
        for name, imgs in self.data.items():
            count = len(imgs)
            summary[name] = count
            total_samples += count
        return summary, total_samples
