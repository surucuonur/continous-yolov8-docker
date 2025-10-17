"""
YOLOv8 Inference Script - Simplified Version
Clean separation of Model and FileWatcher classes for easy building

#TODO:
    1) Model to Json Payload -- DONE
    2) Send Payload to SignalR -- DONE
    3) FileWatcher
    4) Image processing in parallel (with and without GPU) -- DONE
"""
#%%
from ultralytics import YOLO
import argparse
from pathlib import Path
import os
import shutil
from datetime import datetime
import time
import requests
import torch

print("Library loaded successfully")
# print("CUDA available:", torch.cuda.is_available())
# print("PyTorch version:", torch.__version__)
# print("CUDA device count:", torch.cuda.device_count())
# print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# print("CUDA device memory:", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3), "GB")
# print("CUDA device memory free:", torch.cuda.get_device_properties(torch.cuda.current_device()).free_memory / (1024**3), "GB")
# print("CUDA device memory used:", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3) - torch.cuda.get_device_properties(torch.cuda.current_device()).free_memory / (1024**3), "GB")

#%%

class Model:
    """YOLOv8 Model class with GPU/CPU detection and processing"""
    
    def __init__(self, weights_path, conf_threshold=0.25):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.device = self._detect_device()
        self.model = self._load_model()
        
    def _detect_device(self):
        """Detect and configure the best available device (GPU first, then CPU)"""
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
            return f"cuda:{current_device}"
        else:
            print("💻 CUDA not available, using CPU")
            return "cpu"
    
    def _load_model(self):
        """Load the YOLOv8 model"""
        print(f"🚀 Loading YOLOv8 model from {self.weights_path}...")
        try:
            model = YOLO(self.weights_path)
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def process_file(self, filepaths, output_dir):
        """
        Process images with this logic:
            - If a single file: process with full GPU (if available)
            - If a list of files: process in parallel (batch processing)
        
        Args:
            filepaths (str, Path, or list): Single file path or a list of file paths.
            output_dir (str): Output directory for results.

        Returns:
            results (list): List of results for each file.
        """
        # Normalize filepaths input to always be a list
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]

        if len(filepaths) == 1:
            # Single-file mode: use 100% GPU/CPU for best speed
            filepath = filepaths[0]
            print(f"  Processing single file: {Path(filepath).name} on {self.device} (full power)")
            try:
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
                    verbose=False,
                    device=self.device  # Ensures GPU is used if available
                )

                # Print detection summary
                for result in results:
                    num_detections = len(result.boxes) if result.boxes is not None else 0
                    print(f"  ✓ Processed: {num_detections} detection(s) found")
                    if result.boxes is not None and len(result.boxes) > 0:
                        detections = {}
                        for box in result.boxes:
                            # YOLOv8: box.cls is a tensor/array, so ensure correct extraction
                            cls = int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                            class_name = self.model.names[cls]
                            detections[class_name] = detections.get(class_name, 0) + 1
                        for class_name, count in detections.items():
                            print(f"    - {class_name}: {count}")

                return results

            except Exception as e:
                file_name = Path(filepath).name if hasattr(filepath, 'name') else str(filepath)
                print(f"  ✗ Error processing {file_name}: {str(e)}")
                raise e
        else:
            # Multi-file mode: process in parallel (typically batch with less GPU utilization per stream)
            print(f"  Multiple files detected ({len(filepaths)}). Processing in parallel (batch mode)...")
            try:
                results = self.process_images_parallel(filepaths, output_dir)
                return results
            except Exception as e:
                print(f"  ✗ Error processing multiple files in parallel: {str(e)}")
                raise e

    def create_label_dict(self, results, threshold=0.80):
        """Create dictionary of labels with binary detection status"""
        label_dict = {}
        
        # Process each detection from results
        for r in results:
            # Get boxes and confidences from tensor
            boxes = r.boxes
            
            # Get class names from results
            names = r.names
            
            # Initialize all possible labels to 0
            for class_id, name in names.items():
                if name not in label_dict:
                    label_dict[name] = 0
            
            # Each box contains class_id and confidence
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                
                if confidence >= threshold:
                    # Get label name from class_id and set to 1
                    label_name = names[class_id]
                    label_dict[label_name] = 1

        print(f"📊 PPE Detection Status (threshold={threshold}):")
        for label, status in label_dict.items():
            print(f"   {label}: {'✅ Detected' if status else '❌ Not Detected'}")
            
        payload = {
            "Closed_Case": {
                "Top": {"Expected": 1, "Found": 1},
                "Bottom": {"Expected": 1, "Found": 1},
                "Front": {"Expected": 1, "Found": 1},
                "Back": {"Expected": 1, "Found": 1},
                "Left_Side": {"Expected": 1, "Found": 1},
                "Right_Side": {"Expected": 1, "Found": 1},
                "Empty_Wheel_Well": {"Expected": 1, "Found": label_dict["Empty_Wheel_Well"]},
                "Foam": {"Expected": 1, "Found": label_dict["Foam"]},
                "Handle": {"Expected": 2, "Found": label_dict["Handle"]},
                "Handle_Ribs": {"Expected": 2, "Found": label_dict["Handle_Ribs"]},
                "Latch": {"Expected": 2, "Found": label_dict["Latch"]},
                "Latch_Ribs": {"Expected": 2, "Found": label_dict["Latch_Ribs"]},
                "Wheel_Well_With_Wheel": {"Expected": 2, "Found": label_dict["Wheel_Well_With_Wheel"]},
                "State": 2
            }
        }
        return payload
    
    # def _reorganize_results(self, output_dir):
    #     """Reorganize YOLO output files into proper structure"""
    #     temp_dir = os.path.join(output_dir, "temp")
        
    #     if os.path.exists(temp_dir):
    #         # Create subdirectories
    #         image_dir = os.path.join(output_dir, "image")
    #         label_dir = os.path.join(output_dir, "label")
    #         os.makedirs(image_dir, exist_ok=True)
    #         os.makedirs(label_dir, exist_ok=True)
            
    #         # Move image files
    #         for file in os.listdir(temp_dir):
    #             file_path = os.path.join(temp_dir, file)
    #             if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    #                 shutil.move(file_path, os.path.join(image_dir, file))
            
    #         # Move label files
    #         temp_labels = os.path.join(temp_dir, "labels")
    #         if os.path.exists(temp_labels):
    #             for file in os.listdir(temp_labels):
    #                 shutil.move(
    #                     os.path.join(temp_labels, file),
    #                     os.path.join(label_dir, file)
    #                 )
    #             os.rmdir(temp_labels)
            
    #         # Clean up temp directory
    #         if os.path.exists(temp_dir):
    #             shutil.rmtree(temp_dir)


class FileWatcher:
    """FileWatcher class for monitoring directories (empty for now)
    
    This function watches the Input directory, it contains 3 subdirectories:
    - Station3-1
    - Station3-2
    - Station3-3

    For each subdirectory, it will wait for the same naming of the images to come in.
    The images will be coming in real-time, so we need to process them as they come in.
    Each image from each subdirectory has the following naming format: YYYYMMDD-HHMMSS-NNNNN.Jpeg (NNNNN is the frame number)

    The Filewatcher should cache the latest processed image, based on that if there is a new image, it shoul wait for to see the same images from all 3 subdirectories to come in.
    If the same images from all 3 subdirectories come in, it should process the images in parallel.

    After that, it should wait for the next set of images to come in.
    If there is no processed image in the cache, either you can set the oldest image in all sub folders, or you can expect a manual variable to be set.
    """
    
    def __init__(self):
        pass
    
    # TODO: Implement file watching functionality
    # This class is intentionally left empty for future implementation


    
class FunctionAppConnector:
    """
    Local script to connect to Azure Function App and trigger broadcasts to SignalR
    """
    
    def __init__(self, function_url=None, function_key=None):
        """
        Initialize the connector with function app details
        
        Args:
            function_url: Base URL of your function app (e.g., https://your-function-app.azurewebsites.net)
            function_key: Function key for authentication
        """
        import os
        self.function_url = function_url or os.environ.get("FUNCTION_URL", "https://azu-wu2-funcfrntend-d-01.azurewebsites.net")
        self.function_key = function_key or os.environ.get("PROCESS_FUNCTION_KEY")
        if self.function_key:
            self.function_url = self.function_url.rstrip('/')
            self.broadcast_endpoint = f"{self.function_url}/api/broadcast?code={self.function_key}"
        
    def trigger_broadcast(self, message: str = "Broadcast from local script", data: dict = None) -> dict:
        """
        Trigger the broadcast function which will send data to SignalR
        
        Args:
            message: Optional message to include in the broadcast
            data: Custom data payload to send to SignalR. If not provided, function will fail.
            
        Returns:
            dict: Response from the broadcast function
        """
        if not self.function_key:
            print("❌ Error: PROCESS_FUNCTION_KEY not set!")
            return {"error": "Function key not set"}

        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "message": message,
                "data": data
            }
            
            response = requests.post(
                self.broadcast_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Payload sent successfully!")
                return response.json()
            else:
                error_msg = f"Status {response.status_code}"
                print(f"❌ Failed to send payload: {error_msg}")
                return {"error": error_msg, "details": response.text}
                
        except Exception as e:
            print(f"❌ Failed to send payload: {str(e)}")
            return {"error": str(e)}
    
    def listen_and_broadcast(self, interval: int = 5, max_iterations: int = None, data: dict = None):
        """
        Continuously listen and trigger broadcasts at specified intervals
        
        Args:
            interval: Time between broadcasts in seconds (default: 5)
            max_iterations: Maximum number of broadcasts (None for infinite)
            data: Data payload to send with each broadcast
        """
        print(f"Starting continuous broadcast (interval: {interval}s)")
        
        iteration = 0
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    print(f"Reached maximum iterations ({max_iterations})")
                    break
                
                iteration += 1
                timestamp = datetime.now().isoformat()
                message = f"Broadcast #{iteration} at {timestamp}"
                
                result = self.trigger_broadcast(message, data=data)
                
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            print("\nStopping broadcast (KeyboardInterrupt)")
        except Exception as e:
            print(f"Error in broadcast loop: {e}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to the function app
        
        Returns:
            bool: True if connection is successful
        """
        try:
            health_url = f"{self.function_url}/api/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                print("✓ Connection successful!")
                return True
            else:
                print(f"✗ Connection failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False

def main():
    """Main function demonstrating usage of Model and FileWatcher classes"""
    parser = argparse.ArgumentParser(description='YOLOv8 simplified inference')
    parser.add_argument('--weights', type=str, default='/app/weights/best.pt',
                        help='Path to weights file (default: /app/weights/best.pt)')
    parser.add_argument('--source', type=str, default='/app/input',
                        help='Directory to monitor (default: /app/input)')
    parser.add_argument('--conf', type=float, default=0.85,
                        help='Confidence threshold (default: 0.85)')
    parser.add_argument('--output-base', type=str, default='/app/output',
                        help='Base output directory (default: /app/output)')
    
    args = parser.parse_args()
    
    # Check if weights file exists
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Error: Weights file '{args.weights}' not found!")
        return
    
    # Create output directory if it doesn't exist
    Path(args.output_base).mkdir(parents=True, exist_ok=True)
    
    # Initialize the Model class
    model = Model(args.weights, args.conf)
    
    # Initialize the FileWatcher class (empty for now)
    file_watcher = FileWatcher()
    
    # Example usage: Process a single file
    source_path = Path(args.source)
    if source_path.exists():
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
        files = [f for f in source_path.iterdir() 
                if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if files:
            print(f"Found {len(files)} file(s) to process")
            for filepath in files:
                # Create output directory for this file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(args.output_base, f"output_{timestamp}_{filepath.stem}")
                os.makedirs(output_dir, exist_ok=True)
                
                # Process the file
                start_time = time.time()
                try:
                    results = model.process_file(filepath, output_dir)
                    end_time = time.time()
                    time_ms = (end_time - start_time) * 1000
                    print(f"⏱️  Time taken to process {filepath.name}: {time_ms:.2f} milliseconds")
                    print(f"  → Output saved to: {output_dir}")
                except Exception as e:
                    print(f"❌ Error processing {filepath.name}: {e}")
        else:
            print("No files found to process.")
    else:
        print(f"⚠️  Source directory '{args.source}' does not exist.")

# if __name__ == "__main__":
    # main()

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

#%%

#### TESTING ####

# Load the environment variables
from dotenv import load_dotenv
load_dotenv('.env')

# Initialize the parameters
weights_path = "./weights/best.pt"
source_path = "./Input"
conf = 0.85
output_base = "./Output"

# Initialize the Model class
model = Model(weights_path, conf)

# Just give one file to the model
filepath = "./Input/Station3-1/20251014-124754-00010.Jpeg"
results = model.process_file(Path(filepath), output_base)
print(results)

# Process the results
label_dict = model.create_label_dict(results, threshold=0.80)
print(label_dict)

# Initialize the FunctionAppConnector class
function_app_connector = FunctionAppConnector()

# Send the payload to SignalR
result = function_app_connector.trigger_broadcast(message="PPE Detection completed", data=label_dict)
print(result)


# %%
