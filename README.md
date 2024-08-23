## Introduction

## Installation

1. Create a conda environment
   ```
   conda create -n <env_name> python=3.10
   ```
2. Activate the conda environment
    ```
    conda activate <env_name>
    ```
3. Install the required packages
    ```
    pip install -r requirements.txt
    ```

## Setup

For setup please check the params.yaml file and change the values accordingly.

   ```yaml
   model_path: "resources/models/yolov8m.pt"
   result_dir: "results/"
   tracker_config: "resources/config/bytetrack.yaml"
   conf: 0.85
   device: "cuda:0"
   iou: 0.85
   img_size: [ 720, 1080 ]
   video_source: 0
   show: True

```

Adjust the values according to your requirements.

### Video Source

1. RTSP Server
    ``` bash
    source: rtsp://ishwor:subedi@192.168.1.106:5555/h264_opus.sdp
    ```
2. Video File
    ``` bash
    source: path/to/your/file.mp4
    ```
3. Webcam
    ```
   source=0
    ```

## Inference

```
chmod +x start.sh
```

```
./start.sh
```