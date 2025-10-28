"""
YOLOv8 Inference Script - Simplified Version
Clean separation of Model and FileWatcher classes for easy building

#TODO:
    1) Model to Json Payload -- DONE
    2) Send Payload to SignalR -- DONE
    3) FileWatcher -- DONE
    4) Image processing in parallel (with and without GPU) -- DONE

To run the python script: python inference.py --weights=./weights/best.pt --source=./Input --conf=0.85
"""
#%%
from ultralytics import YOLO
import numpy as np
from scipy.stats import mode
import argparse
from pathlib import Path
import os
import shutil
from datetime import datetime, timedelta
import time
import requests
import torch
# Load the environment variables
from dotenv import load_dotenv
load_dotenv('.env')


try:
    print("Library loaded successfully")
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch version:", torch.__version__)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA device memory:", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3), "GB")
    print("CUDA device memory free:", torch.cuda.get_device_properties(torch.cuda.current_device()).free_memory / (1024**3), "GB")
    print("CUDA device memory used:", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3) - torch.cuda.get_device_properties(torch.cuda.current_device()).free_memory / (1024**3), "GB")
except Exception as e:
    print(f"❌ Error loading CUDA libraries: {e}")
#%%

class Model:
    """YOLOv8 Model class with GPU/CPU detection and processing"""
    
    def __init__(self, weights_path, conf_threshold=0.85):
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

    def process_file(self, filepaths):
        """
        Process images with this logic:
            - If a single file: process with full GPU (if available)
            - If a list of files: process in parallel (batch processing)
        
        Args:
            filepaths (str, Path, or list): Single file path or a list of file paths.

        Returns:
            results (list): List of results for each file.
        """
        # Normalize filepaths input to always be a list
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]


        # Single-file mode: use 100% GPU/CPU for best speed
        # Dont save the results anywhere, just return the results
        results = self.model.predict(
            source=filepaths,
            conf=self.conf_threshold,
            save=False,
            save_txt=False,
            save_conf=False,
            show_labels=True,
            show_conf=True,
            line_width=2,
            # project=output_dir,
            # name="temp",
            # exist_ok=True,
            verbose=True,
            device=self.device  # Ensures GPU is used if available
        )

        label_dict = {}
        # Print detection summary
        for i, result in enumerate(results):
            # num_detections = len(result.boxes) if result.boxes is not None else 0
            # print(f"  ✓ Processed: {num_detections} detection(s) found")
            if result.boxes is not None and len(result.boxes) > 0:
                detections = {}
                for box in result.boxes:
                    # YOLOv8: box.cls is a tensor/array, so ensure correct extraction
                    cls = int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                    class_name = self.model.names[cls]
                    detections[class_name] = detections.get(class_name, 0) + 1
                # for class_name, count in detections.items():
                    # print(f"    - {class_name}: {count}")
            # if result is None or (hasattr(result, 'boxes') and (result.boxes is None or len(result.boxes) == 0)):
                # print(f"Station3-{i+1}: No detections found")
            # Create label dictionary for each result
            payload = self.create_payload_json(result)
            label_dict[filepaths[i]] = payload
        return label_dict

    def create_payload_json(self, results, threshold=0.80):
        """Create dictionary of labels with binary detection status"""
        label_dict = {}
        
        if results is None or (hasattr(results, 'boxes') and (results.boxes is None or len(results.boxes) == 0)):
            payload = {
                "Closed_Case": {
                    "Top": {"Expected": 1, "Found": 1},
                    "Bottom": {"Expected": 1, "Found": 1},
                    "Front": {"Expected": 1, "Found": 1},
                    "Back": {"Expected": 1, "Found": 1},
                    "Left_Side": {"Expected": 1, "Found": 1},
                    "Right_Side": {"Expected": 1, "Found": 1},
                    "Empty_Wheel_Well": {"Expected": 1, "Found": 0},
                    "Foam": {"Expected": 1, "Found": 0},
                    "Handle": {"Expected": 2, "Found": 0},
                    "Handle_Ribs": {"Expected": 2, "Found": 0},
                    "Latch": {"Expected": 2, "Found": 0},
                    "Latch_Ribs": {"Expected": 2, "Found": 0},
                    "Wheel_Well_With_Wheel": {"Expected": 2, "Found": 0},
                    "State": 2
                }
            }
            return payload

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

        # print(f"📊 PPE Detection Status (threshold={threshold}):")
        # for label, status in label_dict.items():
            # print(f"   {label}: {'✅ Detected' if status else '❌ Not Detected'}")
            
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

    def results_post_processing(self, results):
        """
        For the 3 angle camerage view, if the voting is greater than 2, then the result is valid.
        """
        final_results_counter = {}

        # Count the number of valid results for each subkey
        for key, payload in results.items():
            for subkey, subvalue in payload["Closed_Case"].items():
                # Check if the subkey is inthe final_results_counter
                if subkey in final_results_counter:
                    # print("Adding to existing subkey")
                    # print(f"Adding new subkey: {subkey}")
                    # print(f"Value: {subvalue['Found']}")
                    final_results_counter[subkey].append(subvalue['Found'])
                else:
                    if subkey != "State":
                        # print("Initializing new subkey")
                        # print(f"Adding new subkey: {subkey}")
                        # print(f"Value: {subvalue['Found']}")

                        final_results_counter[subkey] = [subvalue['Found']]

        # take the median as the final values, and assign it to the fina_result_dict
        final_result_dict = {}
        for key, value in final_results_counter.items():
            final_result_dict[key] = int(np.max(value))
        # print(final_result_dict)


        # # if there are no detections at all,
        # if all(value == 0 for value in final_result_dict.values()):
        #     final_payload = {
        #     "Closed_Case": {
        #         "Top": {"Expected": 1, "Found": 1},
        #         "Bottom": {"Expected": 1, "Found": 1},
        #         "Front": {"Expected": 1, "Found": 1},
        #         "Back": {"Expected": 1, "Found": 1},
        #         "Left_Side": {"Expected": 1, "Found": 1},
        #         "Right_Side": {"Expected": 1, "Found": 1},
        #         "Empty_Wheel_Well": {"Expected": 1, 0},
        #         "Foam": {"Expected": 1, "Found": 0},
        #         "Handle": {"Expected": 2, "Found": 0},
        #         "Handle_Ribs": {"Expected": 2, "Found": 0},
        #         "Latch": {"Expected": 2, "Found": 0},
        #         "Latch_Ribs": {"Expected": 2, "Found": 0},
        #         "Wheel_Well_With_Wheel": {"Expected": 2, "Found": 0},
        #         "State": 2
        #     }
        # }

        final_payload = {
            "Closed_Case": {
                "Top": {"Expected": 1, "Found": 1},
                "Bottom": {"Expected": 1, "Found": 1},
                "Front": {"Expected": 1, "Found": 1},
                "Back": {"Expected": 1, "Found": 1},
                "Left_Side": {"Expected": 1, "Found": 1},
                "Right_Side": {"Expected": 1, "Found": 1},
                "Empty_Wheel_Well": {"Expected": 1, "Found": final_result_dict["Empty_Wheel_Well"]},
                "Foam": {"Expected": 1, "Found": final_result_dict["Foam"]},
                "Handle": {"Expected": 2, "Found": final_result_dict["Handle"]},
                "Handle_Ribs": {"Expected": 2, "Found": final_result_dict["Handle_Ribs"]},
                "Latch": {"Expected": 2, "Found": final_result_dict["Latch"]},
                "Latch_Ribs": {"Expected": 2, "Found": final_result_dict["Latch_Ribs"]},
                "Wheel_Well_With_Wheel": {"Expected": 2, "Found": final_result_dict["Wheel_Well_With_Wheel"]},
                "State": 2
            }
        }
        return final_payload


class FileWatcher:
    """FileWatcher class for monitoring directories
    
    This function watches the Input directory, it contains 3 subdirectories:
    - Station3-1
    - Station3-2
    - Station3-3

    For each subdirectory, it will wait for the same naming of the images to come in.
    The images will be coming in real-time, so we need to process them as they come in.
    Each image from each subdirectory has the following naming format: YYYYMMDD-HHMMSS-NNNNN.Jpeg (NNNNN is the frame number)

    The Filewatcher should cache the latest processed image, based on that if there is a new image, it should wait for to see the same images from all 3 subdirectories to come in.
    If the same images from all 3 subdirectories come in, it should process the images in parallel.

    After that, it should wait for the next set of images to come in.
    If there is no processed image in the cache, either you can set the oldest image in all sub folders, or you can expect a manual variable to be set.
    """
    
    def __init__(self, input_dir, subdirs=["Station3-1", "Station3-2", "Station3-3"], poll_interval=0.1):
        """
        Initialize the FileWatcher
        
        Args:
            input_dir (str or Path): Base input directory containing subdirectories
            subdirs (list): List of subdirectory names to monitor (default: Station3-1, Station3-2, Station3-3)
            poll_interval (float): Time in seconds between directory checks (default: 1.0)
        """
        self.input_dir = Path(input_dir)
        self.subdirs = subdirs
        self.poll_interval = poll_interval
        
        # Cache for the latest processed image identifier
        self.last_processed_identifier = self.get_latest_common_identifier_in_subdirs()
        
        # Validate input directory and subdirectories
        if not self.input_dir.exists():
            raise ValueError(f"Input directory '{input_dir}' does not exist!")
        
        for subdir in self.subdirs:
            subdir_path = self.input_dir / subdir
            if not subdir_path.exists():
                raise ValueError(f"Subdirectory '{subdir}' does not exist in {input_dir}!")
        
        print(f"📁 FileWatcher initialized")
        print(f"   Input directory: {self.input_dir}")
        print(f"   Monitoring subdirectories: {', '.join(self.subdirs)}")
        # print the 5 files from each subdirectory
        for subdir in self.subdirs:
            subdir_path = self.input_dir / subdir
            images = [f for f in subdir_path.iterdir() if f.is_file()]
            images_sorted = sorted(images)
            print(f"   {subdir}: {len(images_sorted)} files")
            for img in images_sorted[-5:]:
                print(f"      {img.name}")
        print(f"   Poll interval: {self.poll_interval}s")
    
    def extract_identifier(self, filename):
        """
        Extract the unique identifier from an image filename.
        Format: YYYYMMDD-HHMMSS-NNNNN.Jpeg
        Returns: YYYYMMDD-HHMMSS-NNNNN (without extension)
        
        Args:
            filename (str): The filename to extract identifier from
            
        Returns:
            str: The identifier or None if format doesn't match
        """
        if isinstance(filename, Path):
            filename = filename.name
        
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Validate format: YYYYMMDD-HHMMSS-NNNNN
        parts = name_without_ext.split('-')
        if len(parts) == 3 and len(parts[0]) == 8 and len(parts[1]) == 6 and len(parts[2]) == 5:
            return name_without_ext
        
        return None
    
    def get_all_images_in_subdir(self, subdir):
        """
        Get all image files in a specific subdirectory
        
        Args:
            subdir (str): Name of the subdirectory
            
        Returns:
            list: List of Path objects for images in the subdirectory
        """
        subdir_path = self.input_dir / subdir
        valid_extensions = {'.jpg', '.jpeg', '.Jpeg', '.JPG', '.JPEG'}
        
        images = [f for f in subdir_path.iterdir() 
                if f.is_file() and f.suffix in valid_extensions]
        
        images_sorted = sorted(images)[-50:]
        return images_sorted
    
    def get_all_files_in_subdir(self, subdir):
        """
        Get all image identifiers in a specific subdirectory
        
        Args:
            subdir (str): Name of the subdirectory
            
        Returns:
            set: Set of identifiers found in the subdirectory
        """
        images = self.get_all_images_in_subdir(subdir)
        identifiers = set()
        
        for img in images:
            identifier = self.extract_identifier(img.name)
            if identifier:
                identifiers.add(identifier)
        
        return identifiers


    def get_latest_common_identifier_in_subdirs(self):
        """
        Find the latest common identifier (image naming format: YYYYMMDD-HHMMSS-NNNNN) in the last 5 images in all subdirectories.

        Note: Sometimes the image timestamps might have a difference of 1 second, thus we utilized the timestamp for sorting, and check the sequence number for matching.
        """

        # create a dict of identifiers with the subdir name as the key
        files_dict = {subdir: self.get_all_files_in_subdir(subdir) for subdir in self.subdirs}

        # sort each identifier in descending order
        files_sorted_raw = {subdir: sorted(files, reverse=True) for subdir, files in files_dict.items()}

        # Only take the first 20 rows for each subdir
        files_sorted = {subdir: files_sorted_raw[subdir][:20] for subdir, files in files_sorted_raw.items()}


        ############## FIND COMMON BY TIMESTAMP OF THE IMAGE ##############
        def get_timestamp(image_paths):
            image_paths_timestamp = []
            for image_path in image_paths:
                # Now the naming look like this: YYYYMMDD-HHMMSS
                timestamp = image_path.split('-')[1]
                image_paths_timestamp.append(int(timestamp))
            return image_paths_timestamp

        files_sorted_timestamp = {subdir: get_timestamp(files) for subdir, files in files_sorted.items()}
        last_files = {subdir: files[0] for subdir, files in files_sorted_timestamp.items()}
        min_timestamp = min(last_files.values())
        # print(f"Minimum timestamp: {min_timestamp}")

        ############## FIND COMMON BY TIMESTAMP OF THE IMAGE ##############
        latest_common_file = {}
        for subdir, timestamp_list in files_sorted_timestamp.items():

            # IF the length of the timestamp list is less than 3, wait for 6 seconds and try again
            if len(timestamp_list) < 3:
                print(f"Waiting for more images to come in...")
                time.sleep(6)
                return self.get_latest_common_identifier_in_subdirs()
            
            else:
                timestamps_array = np.array(timestamp_list)
                timestamps_array_delta = timestamps_array - min_timestamp

                # find the element index that is closes to the zero
                arg_min = np.argmin(np.abs(timestamps_array_delta))
                latest_common_file[subdir] = files_sorted[subdir][arg_min]
        return latest_common_file
        
    
    def get_image_paths_for_identifier(self, identifier):
        image_paths = []
        
        for subdir in self.subdirs:
            images = self.get_all_images_in_subdir(subdir)
            for img in images:
                if self.extract_identifier(img.name) == identifier:
                    image_paths.append(img)
                    break
        
        return image_paths
    
    def watch_incoming_images(self):
        """
        This function watches the input subdirectories untill a new common identifier is found.
        Once found, it stops the loop, and returns the path of each identifier in each subdirectory as a list.
        """
        print("\n\n\n")
        print("--------------------------------"*2)
        print(f"Watching for new common identifier in the input subdirectories")
        print(f"Last processed identifier: {self.last_processed_identifier}")
        print(f"Sleep interval: {self.poll_interval}s")
        while True:
            latest_common_identifier = self.get_latest_common_identifier_in_subdirs()
            if (latest_common_identifier != self.last_processed_identifier) and (latest_common_identifier is not None):
                print(f"New common files found: {latest_common_identifier}")
                print(f"Last processed identifier: {self.last_processed_identifier}")
                # return the path of each identifier in each subdirectory as a list
            # image_paths_dict = {subdir: self.get_image_paths_for_identifier(latest_common_identifier[subdir]) for subdir in self.subdirs}
                image_paths = []
                for subdir, identifier in latest_common_identifier.items():
                    image_path = f'{self.input_dir}/{subdir}/{identifier}.Jpeg'
                    image_paths.append(image_path)
                    self.last_processed_identifier = latest_common_identifier
                return image_paths
                
            else:
                pass
            time.sleep(self.poll_interval)

    def get_status(self):
        """
        Get current status of the FileWatcher
        
        Returns:
            dict: Status information
        """
        common_identifiers = self.find_common_identifiers()
        pending = []
        
        if self.last_processed_identifier:
            pending = [id for id in common_identifiers if id > self.last_processed_identifier]
        else:
            pending = common_identifiers
        
        return {
            "last_processed": self.last_processed_identifier,
            "pending_count": len(pending),
            "pending_identifiers": pending[:5],  # Show first 5
            "total_common": len(common_identifiers)
        }



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

class timer:
    """
    Timer class to time the execution of a function
    """
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        # print(f"[⏱️] Time taken for {self.func.__name__}: {elapsed_ms:.2f} ms")



def print_processed_images(processed_images, image_paths):
        ###### PRINT THE PROCESSED IMAGES ######
        processed_images["Station3-1"].append(image_paths[0].split("/")[-1])
        processed_images["Station3-2"].append(image_paths[1].split("/")[-1])
        processed_images["Station3-3"].append(image_paths[2].split("/")[-1])
        if len(processed_images["Station3-1"]) > 10:
            processed_images["Station3-1"] = processed_images["Station3-1"][-10:]
        if len(processed_images["Station3-2"]) > 10:
            processed_images["Station3-2"] = processed_images["Station3-2"][-10:]
        if len(processed_images["Station3-3"]) > 10:
            processed_images["Station3-3"] = processed_images["Station3-3"][-10:]
        print("--------------------------------"*2)
        print("Processed images:")
        print(f"Station3-1: {processed_images['Station3-1']}")
        print(f"Station3-2: {processed_images['Station3-2']}")
        print(f"Station3-3: {processed_images['Station3-3']}")
        print("--------------------------------"*2)

def main():
    """Main function demonstrating usage of Model and FileWatcher classes"""
    parser = argparse.ArgumentParser(description='YOLOv8 simplified inference')
    parser.add_argument('--weights', type=str, default='/app/weights/best.pt',
                        help='Path to weights file (default: /app/weights/best.pt)')
    parser.add_argument('--source', type=str, default='/app/input',
                        help='Directory to monitor (default: /app/input)')
    parser.add_argument('--conf', type=float, default=0.85,
                        help='Confidence threshold (default: 0.85)')
    # parser.add_argument('--output-base', type=str, default='/app/output',
                        # help='Base output directory (default: /app/output)')
    
    args = parser.parse_args()



    # Initialize the parameters
    weights_path = args.weights
    input_path = args.source
    conf = args.conf
    # output_base = os.environ.get("OUTPUT_BASE")

    # # Initialize the Model class
    print(f"Initializing Model with weights: {weights_path} and confidence threshold: {conf}")
    model = Model(weights_path, conf)

    # Initialize the FunctionAppConnector class
    print(f"Initializing FunctionAppConnector")
    function_app_connector = FunctionAppConnector()

    # # Initialize the FileWatcher class
    print(f"Initializing FileWatcher with input path: {input_path} and subdirectories: Station3-1, Station3-2, Station3-3")
    file_watcher = FileWatcher(input_path, subdirs=["Station3-1", "Station3-2", "Station3-3"], poll_interval=0.1)

    def timed(label):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                elapsed = (end - start)
                return result, elapsed
            return wrapper
        return decorator

    @timed("file_watcher.watch_incoming_images")
    def timed_watch_images():
        return file_watcher.watch_incoming_images()

    @timed("model.process_file")
    def timed_process_file(image_paths):
        return model.process_file(image_paths)

    @timed("model.results_post_processing")
    def timed_post_processing(model, results):
        # Re-initialize model just like the original code
        return model.results_post_processing(results)

    @timed("function_app_connector.trigger_broadcast")
    def timed_broadcast(function_app_connector, final_payload):
        return function_app_connector.trigger_broadcast(message="PPE Detection completed", data=final_payload)


    processed_images = {"Station3-1": [], "Station3-2": [], "Station3-3": []}   # Cache for printing the processed images
    total_execution_times = []
    while True:
        times = []
        image_paths, elapsed = timed_watch_images()
        times.append(("file_watcher.watch_incoming_images", elapsed))

        results, elapsed = timed_process_file(image_paths)
        times.append(("model.process_file", elapsed))

        final_payload, elapsed = timed_post_processing(model, results)
        # print(f"✅ Final payload created: {final_payload}")
        times.append(("model.results_post_processing", elapsed))

        result, elapsed = timed_broadcast(function_app_connector, final_payload)
        print(f"✅ Payload sent successfully! {result}")
        times.append(("function_app_connector.trigger_broadcast", elapsed))

        ###### PRINT THE EXECUTION TIMES ######
        print("[⏱️] Execution times (s):")
        for label, t in times:
            print(f"   {label}: {t:.2f} s")
        print(f"Total time: {sum(t for _, t in times):.2f} s")
        total_execution_times.append(sum(t for _, t in times))
        if len(total_execution_times) > 100:
            total_execution_times = total_execution_times[-100:]
        print(f"Last 10 total execution times: {total_execution_times[-10:]}")
        print(f"Average total execution time (last 100 iterations): {sum(total_execution_times) / len(total_execution_times):.2f} s")
        print("Current Time:", datetime.now().isoformat(timespec='seconds'))

        ###### PRINT THE PROCESSED IMAGES ######
        print_processed_images(processed_images, image_paths)


#%%
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

#%%

#### TESTING ####

# # Load the environment variables
# from dotenv import load_dotenv
# load_dotenv('.env')

# # Initialize the parameters
# weights_path = "./weights/best.pt"
# source_path = "./Input"
# conf = 0.85
# output_base = "./Output"

# # # Initialize the Model class
# model = Model(weights_path, conf)

# # Initialize the FunctionAppConnector class
# function_app_connector = FunctionAppConnector()

# # # Initialize the FileWatcher class
# file_watcher = FileWatcher(source_path, subdirs=["Station3-1", "Station3-2", "Station3-3"], poll_interval=1.0)



# while True:
#     # Time file_watcher.watch_incoming_images()
#     start_time = time.time()
#     image_paths = file_watcher.watch_incoming_images()
#     end_time = time.time()
#     elapsed_ms = (end_time - start_time) * 1000
#     print(f"[⏱️] Time taken for file_watcher.watch_incoming_images(): {elapsed_ms:.2f} ms")
#     # print(image_paths)

#     # Time model.process_file
#     start_time = time.time()
#     results = model.process_file(image_paths)
#     end_time = time.time()
#     elapsed_ms = (end_time - start_time) * 1000
#     print(f"[⏱️] Time taken for model.process_file(): {elapsed_ms:.2f} ms")
#     # print(results)

#     # Time model.results_post_processing
#     start_time = time.time()
#     model = Model(weights_path, conf)
#     final_payload = model.results_post_processing(results)
#     end_time = time.time()
#     elapsed_ms = (end_time - start_time) * 1000
#     print(f"[⏱️] Time taken for model.results_post_processing(): {elapsed_ms:.2f} ms")
#     # print(final_payload)

#     # Time function_app_connector.trigger_broadcast
#     start_time = time.time()
#     result = function_app_connector.trigger_broadcast(message="PPE Detection completed", data=final_payload)
#     end_time = time.time()
#     elapsed_ms = (end_time - start_time) * 1000
#     print(f"[⏱️] Time taken for function_app_connector.trigger_broadcast(): {elapsed_ms:.2f} ms")
#     # print(result)
# # %%

# # %%
