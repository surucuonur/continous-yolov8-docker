"""
YOLOv8 Inference Script - Docker Version with Auto-Detection
Monitors input directory and automatically processes new images/videos
Modified for Docker container usage with volume mounts
Enhanced with CUDA GPU detection and usage
"""
from ultralytics import YOLO
import argparse
from pathlib import Path
import os
import shutil
from datetime import datetime
import time
import hashlib
import json
import signal
import sys
from threading import Thread, Event
import time
import torch

def check_cuda_availability():
    """Check if CUDA is available and return device info"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
        
        print(f"🚀 CUDA GPU detected!")
        print(f"   Device: {device_name}")
        print(f"   Device ID: {current_device}")
        print(f"   Total Memory: {memory_total:.1f} GB")
        print(f"   Available Devices: {device_count}")
        return True, f"cuda:{current_device}"
    else:
        print("💻 CUDA not available, using CPU")
        return False, "cpu"

class FileWatcher:
    """Monitors directory for new files and processes them automatically"""
    
    def __init__(self, model, watch_dir, output_base, conf_threshold=0.25, poll_interval=1.0, device="cpu"):
        self.model = model
        self.watch_dir = Path(watch_dir)
        self.output_base = output_base
        self.conf_threshold = conf_threshold
        self.poll_interval = poll_interval
        self.device = device
        self.processed_files = self.load_processed_files()
        self.stop_event = Event()
        
        # Move model to device if CUDA is available
        if device != "cpu":
            print(f"📱 Moving model to {device}")
            self.model.to(device)
        
    def load_processed_files(self):
        """Load list of already processed files"""
        cache_file = Path(self.output_base) / '.processed_files.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()
    
    def save_processed_files(self):
        """Save list of processed files"""
        cache_file = Path(self.output_base) / '.processed_files.json'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(list(self.processed_files), f)
    
    def get_file_hash(self, filepath):
        """Generate hash of file to detect changes"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def get_file_id(self, filepath):
        """Create unique identifier for file"""
        stat = os.stat(filepath)
        return f"{filepath}_{stat.st_size}_{stat.st_mtime}"
    
    def is_file_stable(self, filepath, stability_time=0.5):
        """Check if file has stopped being written to"""
        try:
            initial_size = os.path.getsize(filepath)
            time.sleep(stability_time)
            final_size = os.path.getsize(filepath)
            return initial_size == final_size
        except:
            return False
    
    def process_file(self, filepath):
        """Process a single file"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New file detected: {filepath.name}")
        
        # Wait for file to be completely written
        print(f"  Waiting for file to stabilize...")
        max_wait = 10
        waited = 0
        while not self.is_file_stable(filepath) and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        
        # Create timestamp for this specific inference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories for this file
        base_dir = os.path.join(self.output_base, f"output_{timestamp}_{filepath.stem}")
        output_dir = os.path.join(base_dir, "output")
        label_dir = os.path.join(output_dir, "label")
        image_dir = os.path.join(output_dir, "image")
        
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        try:
            # Run inference with device specification
            print(f"  Processing: {filepath.name} on {self.device}")
            results = self.model.predict(
                source=str(filepath),
                conf=self.conf_threshold,
                save=True,
                save_txt=True,
                save_conf=True,
                show_labels=True,
                show_conf=True,
                line_width=2,
                project=output_dir,
                name="temp",
                exist_ok=True,
                verbose=False,  # Suppress YOLO output
                device=self.device  # Use specified device
            )
            
            # Reorganize output files
            self.reorganize_results(output_dir, image_dir, label_dir)
            
            # Print detection summary
            for result in results:
                num_detections = len(result.boxes) if result.boxes is not None else 0
                print(f"  ✓ Processed: {num_detections} detection(s) found")
                
                # Print detected classes
                if result.boxes is not None and len(result.boxes) > 0:
                    detections = {}
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        detections[class_name] = detections.get(class_name, 0) + 1
                    
                    for class_name, count in detections.items():
                        print(f"    - {class_name}: {count}")
            
            print(f"  → Output saved to: {base_dir}")
            
            # Mark file as processed
            file_id = self.get_file_id(filepath)
            self.processed_files.add(file_id)
            self.save_processed_files()
            
        except Exception as e:
            print(f"  ✗ Error processing {filepath.name}: {str(e)}")
    
    def reorganize_results(self, output_dir, image_dir, label_dir):
        """Reorganize YOLO output files"""
        temp_dir = os.path.join(output_dir, "temp")
        
        if os.path.exists(temp_dir):
            # Move image files
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    shutil.move(file_path, os.path.join(image_dir, file))
            
            # Move label files
            temp_labels = os.path.join(temp_dir, "labels")
            if os.path.exists(temp_labels):
                for file in os.listdir(temp_labels):
                    shutil.move(
                        os.path.join(temp_labels, file),
                        os.path.join(label_dir, file)
                    )
                os.rmdir(temp_labels)
            
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def scan_directory(self):
        """Scan directory for new files"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
        
        if not self.watch_dir.exists():
            return []
        
        new_files = []
        for filepath in self.watch_dir.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
                file_id = self.get_file_id(filepath)
                if file_id not in self.processed_files:
                    new_files.append(filepath)
        
        return new_files
    
    def watch(self):
        """Main watching loop"""
        print(f"\n🔍 Monitoring directory: {self.watch_dir}")
        print(f"📊 Confidence threshold: {self.conf_threshold}")
        print(f"📁 Output directory: {self.output_base}")
        print(f"🖥️  Processing device: {self.device}")
        print(f"\n⏳ Waiting for new images/videos... (Press Ctrl+C to stop)\n")
        
        # Process any existing files first
        initial_files = self.scan_directory()
        if initial_files:
            print(f"Found {len(initial_files)} unprocessed file(s) in directory")
            for filepath in initial_files:
                if self.stop_event.is_set():
                    break
                start_time = time.time()
                self.process_file(filepath)
                end_time = time.time()
                time_ms = (end_time - start_time) * 1000
                print(f"⏱️  Time taken to process {filepath.name}: {time_ms:.2f} milliseconds")
        
        # Monitor for new files
        while not self.stop_event.is_set():
            try:
                new_files = self.scan_directory()
                for filepath in new_files:
                    if self.stop_event.is_set():
                        break
                    start_time = time.time()
                    self.process_file(filepath)
                    end_time = time.time()
                    time_ms = (end_time - start_time) * 1000
                    print(f"⏱️  Time taken to process {filepath.name}: {time_ms:.2f} milliseconds")
                
                time.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in watch loop: {e}")
                time.sleep(self.poll_interval)
        
        print("\n👋 Stopping file watcher...")
    
    def stop(self):
        """Stop the watcher"""
        self.stop_event.set()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n📛 Received shutdown signal, cleaning up...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 auto-inference with file monitoring')
    parser.add_argument('--weights', type=str, default='/app/weights/best.pt',
                        help='Path to weights file (default: /app/weights/best.pt)')
    parser.add_argument('--source', type=str, default='/app/input',
                        help='Directory to monitor (default: /app/input)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output-base', type=str, default='/app/output',
                        help='Base output directory (default: /app/output)')
    parser.add_argument('--poll-interval', type=float, default=1.0,
                        help='Polling interval in seconds (default: 1.0)')
    parser.add_argument('--single-run', action='store_true',
                        help='Process existing files once and exit (no monitoring)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cpu, cuda, or cuda:0 (default: auto)')
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check device availability
    if args.device == 'auto':
        cuda_available, device = check_cuda_availability()
    elif args.device.startswith('cuda'):
        if torch.cuda.is_available():
            device = args.device
            print(f"🚀 Using specified CUDA device: {device}")
        else:
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device = "cpu"
    else:
        device = args.device
        print(f"💻 Using device: {device}")
    
    # Check if weights file exists
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Error: Weights file '{args.weights}' not found!")
        print("Make sure to mount your weights directory properly.")
        return
    
    # Check if source directory exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"⚠️  Warning: Source directory '{args.source}' does not exist yet.")
        print("Creating directory and waiting for files...")
        source_path.mkdir(parents=True, exist_ok=True)
    
    # Load the model once
    print(f"🚀 Loading YOLOv8 model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create output directory if it doesn't exist
    Path(args.output_base).mkdir(parents=True, exist_ok=True)
    
    if args.single_run:
        # Single run mode - process existing files and exit
        print("\n📋 Running in single-run mode (no monitoring)")
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
        files = [f for f in source_path.iterdir() 
                if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if not files:
            print("No files found to process.")
        else:
            print(f"Found {len(files)} file(s) to process")
            watcher = FileWatcher(model, args.source, args.output_base, 
                                args.conf, args.poll_interval, device)
            for filepath in files:
                start_time = time.time()
                watcher.process_file(filepath)
                end_time = time.time()
                time_ms = (end_time - start_time) * 1000
                print(f"⏱️  Time taken to process {filepath.name}: {time_ms:.2f} milliseconds")
    else:
        # Continuous monitoring mode
        watcher = FileWatcher(model, args.source, args.output_base, 
                            args.conf, args.poll_interval, device)
        try:
            watcher.watch()
        except KeyboardInterrupt:
            print("\n✋ Interrupted by user")
        finally:
            watcher.stop()
            print("🏁 File watcher stopped")

if __name__ == "__main__":
    main()

'''
docker run --rm -it \
  -v $(pwd)/weights:/app/weights:ro \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  yolov8-inference:latest \
  --weights /app/weights/best.pt \
  --source /app/input \
  --conf 0.78  
'''