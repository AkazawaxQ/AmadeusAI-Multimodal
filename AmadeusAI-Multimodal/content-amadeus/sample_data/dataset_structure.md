# ASL Dataset Structure

This document describes the structure of the custom ASL hand sign dataset
used to train the YOLOv11 model in the Amadeus project.

## Dataset Overview
- Number of Data: 3702 images
- Dataset type: Custom ASL hand sign images
- Data source: Self-collected images
- Annotation format: YOLO
- Classes: ASL alphabet + control gestures (Enter, Delete, Space)

## Directory Structure
ProjectAmadeusASLDataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

## Notes
- Images were captured manually under different lighting and background conditions
- Dataset includes control gestures used for UI interaction
- Full dataset is not publicly included due to size and ownership
