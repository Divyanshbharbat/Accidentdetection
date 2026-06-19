import os
import shutil
from ultralytics import YOLO

def main():
    # Define dataset path
    dataset_path = r"C:\Users\bharb\Downloads\archive (1)\data"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        print("Please double check the directory path.")
        return
        
    print(f"Dataset directory found at: {dataset_path}")
    
    # Check split directories
    train_dir = os.path.join(dataset_path, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Cannot find 'train' folder in {dataset_path}")
        return

    print("Step 1: Loading pretrained YOLOv8 classification model (yolov8n-cls.pt)...")
    # Load pretrained YOLO classification model
    model = YOLO("yolov8n-cls.pt")
    
    print("\nStep 2: Starting model training...")
    print("Training configuration:")
    print(" - Dataset:", dataset_path)
    print(" - Epochs: 10")
    print(" - Image Size: 224")
    print(" - Batch Size: 16")
    print(" - Workers: 0 (Optimized for Windows)")
    
    try:
        # Run training
        # We use imgsz=224 for standard image classification, which runs much faster
        results = model.train(
            data=dataset_path,
            epochs=10,
            imgsz=224,
            batch=16,
            workers=0,
            verbose=True
        )
        
        print("\nStep 3: Training completed successfully!")
        
        # Check and copy weights
        # Usually weights are saved in runs/classify/train/weights/best.pt
        # If there are multiple runs, they are saved in runs/classify/train2, train3, etc.
        # Ultralytics results.save_dir contains the actual path of the current run.
        actual_save_dir = getattr(results, "save_dir", "runs/classify/train")
        best_weights_path = os.path.join(actual_save_dir, "weights", "best.pt")
        
        if os.path.exists(best_weights_path):
            shutil.copy(best_weights_path, "best.pt")
            print("--------------------------------------------------")
            print("SUCCESS: Copied 'best.pt' to project root!")
            print(f"Saved weights: {os.path.abspath('best.pt')}")
            print("--------------------------------------------------")
        else:
            print(f"Warning: Could not find best.pt at '{best_weights_path}'")
            # Try searching in runs/classify/
            found = False
            for root, dirs, files in os.walk("runs/classify"):
                if "best.pt" in files:
                    src = os.path.join(root, "best.pt")
                    shutil.copy(src, "best.pt")
                    print("--------------------------------------------------")
                    print(f"SUCCESS: Copied '{src}' to project root as 'best.pt'!")
                    print("--------------------------------------------------")
                    found = True
                    break
            if not found:
                print("Could not locate trained weights automatically. Check the 'runs' directory.")
                
    except Exception as e:
        print(f"\nError occurred during training: {e}")

if __name__ == "__main__":
    main()
