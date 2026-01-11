# ***People detection, counting and forecasting using YOLOv8: Smart crowd analyzing AI*** #

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***1 Project overview and development*** ##

This markdown shows the step-by-step development of an AI model that does smart crowd analyzing through video imagery.
The system evolved from detecting people in single images to analyzing a live video stream, turning detections into data, and forecasting future crowdedness. Across the different versions, the focus shifted from pure visual output to data driven insight, while also addressing privacy and usability. Multiple 

--------------------------------------------------------------------------------------------------------------------------------------------------

## **2 Evolution of the system (version overview)** ##
v1: Image based people detection
The first version of the project focused on understanding how object detection works.
A YOLO model was used to detect people in single static images using bounding boxes.
At this stage:

- The model only worked on images, not video
- Each detected person was marked with a rectangle
- The total number of detected people was saved to a CSV file
- The output was mainly visual and used for validation

This version was useful for learning the basics of model inference and result handling, but it had no notion of time or trends.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***2.1 v2: Video processing and time based counting*** ##
In version 2, the system was extended to work with video input instead of images.
Frames were extracted from a video at a reduced frame rate to limit computational load.

New elements introduced:
- Processing multiple frames over time
- Counting people per frame
- Plotting people counts over time
- Basic data logging for later analysis

This version introduced the idea of crowdedness as a time-dependent variable, but detections were still based on bounding boxes.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***2.2 v3: Live streaming and instance segmentation*** ##
Version 3 marked a major step forward by switching from local video files to a live video stream.
At the same time, the model was upgraded from bounding box detection to instance segmentation.
Key improvements:

-Real time processing of a live stream
-Pixel level segmentation of each person
-More accurate detection when people overlap
-Better separation between individuals in crowded scenes

This version significantly improved detection quality and made the system suitable for real-world scenarios.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***2.3 v3.5: Early forecasting and trend analysis*** ##
In version 3.5, the focus shifted from detection to data interpretation.
Historical people counts were used to make the first simple predictions.
What was added:

- Storage of historical counts over longer periods
- Basic trend analysis
- Initial forecasting using simple regression techniques

Although the predictions were still rough, this version introduced the idea that detection data can be used for decision support, not just monitoring.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***2.4 v4: Complete end-to-end system (sub-final version)*** ##
This sub-final version integrates all previous components into a single, structured pipeline.

This version includes:
- Live stream processing with YOLOv8 instance segmentation
- Privacy-aware face blurring to reduce identification risks
- Conversion of detections into structured, time-based data
- Noise reduction through aggregation and smoothing
- Forecasting crowdedness up to 12 hours ahead
- Automatic saving of datasets and visual reports

Version 4 represents a full transition from computer vision output to data-driven insight, making the system suitable for analysis, reporting, and future extension.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***2.5 v5: Video-based forecasting with advanced time-series modeling*** ##
Version 5 focused on improving the forecasting quality and system robustness by switching from simple regression to a dedicated time-series forecasting model.

Key changes introduced in v5:
- Processing full video files instead of live streams
- Frame sampling to reduce computational cost
- Instance segmentation–based blurring of complete individuals
- Use of Facebook Prophet for time-series forecasting
- Forecasting crowdedness up to 6 hours ahead
- Automatic generation of processed video output
-Improved visual reporting of forecasts

This version represents the transition from basic trend estimation to statistically grounded time-series forecasting, making predictions more stable and realistic over longer horizons.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***2.6 v6: Frame-based large-scale analysis and refined forecasting (final version)*** ##
Version 6 is the most refined and structured version of the system.
Instead of processing live streams or videos directly, the system operates on pre-extracted image frames.
This allows full control over temporal resolution, reproducibility, and large-scale analysis.

Major improvements in v6 include:
- Processing thousands of individual frames as a dataset
- Confidence-based person detection
- Fully structured minute-level aggregation
- Custom seasonal patterns in Prophet
- Long-horizon forecasting with confidence intervals
- Clear separation between observed data and predictions
- Publication-ready visualizations

This version shifts the system from a real-time tool to a data-centric analytical pipeline, suitable for research, reporting, and further optimization.

--------------------------------------------------------------------------------------------------------------------------------------------------

## **3 Short non technical explanation of how this AI model works** ##

The AI model looks at each video frame and tries to recognize people.
Instead of drawing simple rectangles, it identifies the exact shape of each person.
This allows the system to count people more reliably, even when they overlap or stand close together.
The system does not store images; it only keeps numerical summaries such as how many people were present at a certain moment.
By looking at how these numbers change over time, the system can make a reasonable prediction about future crowdedness.

--------------------------------------------------------------------------------------------------------------------------------------------------

## ***4 Code structure and explanation*** ##

### ***4.1 Model loading and configuration*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from pathlib import Path

model = YOLO("yolov8n-seg.pt")
```

In this block all required libraries are imported and initialization of the YOLOv8 instance segmentation model is being established.
The model is pre trained on the coco dataset and can detect people at pixel level instead of using only bounding boxes.

--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.2 Video stream input and runtime control*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
STREAM_URL = "https://test-streams.mux.dev/test_001/stream.m3u8"

TARGET_FPS = 1
MAX_RUNTIME = 300
```
Here is the source of the video that is used on the model. Also, how often frames are processed and how long the system is allowed to run is defined here.

```python
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise Exception("Stream cannot be opened")
```
Here the stream is openend and getting checked wheter the frames can be processed or not.

--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.3 Privacy compliance through face blurring*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
def blur_face(frame, mask):
    """
    Blurring the upper part of a person mask
    to comply with privacy (GDPR).
    """
    h, w = frame.shape[:2]
    mask = cv2.resize(mask, (w, h))

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return frame

    top = ys.min()
    bottom = top + int(0.3 * (ys.max() - top))
    left, right = xs.min(), xs.max()

    roi = frame[top:bottom, left:right]
    roi = cv2.GaussianBlur(roi, (31, 31), 0)
    frame[top:bottom, left:right] = roi

    return frame
```

This function estimates the face region using the upper part of the person maks and applies a blur. This will cause the individuals to not be identified, and this way comply with GDPR.

--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.4 Live detection and people counting*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(1)
        continue

    elapsed = time.time() - start_time
    if elapsed > MAX_RUNTIME:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = model.predict(
        frame_rgb,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]
```

This loop continously reads frames from the stream and runs instance segmentation on each frame

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
people_count = 0
annotated = frame_rgb.copy()

if result.masks is not None:
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for i, cls in enumerate(classes):
        if cls == 0:
            people_count += 1
            annotated = blur_face(annotated, masks[i])
```

Each detected person (coco class 0) is counted and processed with face blurring before being logged.

--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.5 Creating structured data from detections*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
records.append({
    "time_s": elapsed,
    "people": people_count
})
```

At this point, visual detections are converted into structured numerical data. This is where raw video information becomes usable data for analysis and forecasting.

--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.6 Data preparation and noise reduction*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
df = pd.DataFrame(records)

df["minute"] = (df["time_s"] // 60).astype(int)
df_min = df.groupby("minute").mean().reset_index()

df_min["people_smooth"] = df_min["people"].rolling(
    window=3,
    min_periods=1
).mean()
```

The raw second based data is aggregated per minute and smoothed. This reduces short term fluctuations and creates a more stable signal for forecasting.

--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.7 Forecasting future crowdedness*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
X = df_min["minute"].values.reshape(-1, 1)
y = df_min["people_smooth"].values

reg = LinearRegression()
reg.fit(X, y)
```

A simple linear regression model is trained using historical people counts.

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
future_minutes = np.array([15, 30, 60, 180, 360, 720]).reshape(-1, 1)
future_preds = reg.predict(future_minutes)
```

The model predicts expected crowdedness up to 12 hours into the future.


--------------------------------------------------------------------------------------------------------------------------------------------------

### ***4.8 Saving data and output*** ###

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
DATA_DIR = Path("../data/new_data")
VIS_DIR = Path("../visualization/graphs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
```

Here the folders are automatically created to store results.

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
df_min.to_csv(DATA_DIR / "people_count.csv", index=False)
df_forecast.to_csv(DATA_DIR / "people_forecast.csv", index=False)
```

Processed data and forecasts are saved as CSV files.

--------------------------------------------------------------------------------------------------------------------------------------------------

```python
plt.savefig(VIS_DIR / "crowdedness_forecast.png")
```

A visualization combining current and predicted crowdedness is saved for reporting or analysis.

--------------------------------------------------------------------------------------------------------------------------------------------------