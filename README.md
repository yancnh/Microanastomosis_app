# Microanastomosis Training App
## Demo

https://github.com/user-attachments/assets/3420876d-dc08-4cb5-8457-c30764820aa8



## Functions
The app connects object tracking, action segmentaton and LLM modelsthrough a thin FastAPI process:

- `microanastomosis_tool`: YOLO + DeepSORT instrument detection/tracking
- `actsegformer`: action segmentation
- 'llm': interaction context chat
- `vision_api`: FastAPI 

## Current behavior

- Detection and tracking use the real fine-tuned microanastomosis_tool checkpoint.
- Instrument tip/trail extraction and tracking.
- Action probabilities use the real 7-class ActSegFormer checkpoint:
  No action, Vessel cutting, Needle handling, Needle touching vessel,
  Needle withdrawing, Knot tying, and Knot cutting.
- Chat is a worker grounded in the current tracks, action, and score.
- The displayed performance score is from supervised training model with experts scoring.

