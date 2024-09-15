import gradio as gr  # this facililtates the user interface
from ultralytics import YOLO
import os

HOME = os.getcwd()


def detect(image):
    model = YOLO(f"{HOME}/models/yolov10s.pt")
    results = model(image)[0]
    result_file_path = "result.jpg"
    results.save(filename=result_file_path)
    return result_file_path


demo = gr.Interface(
    fn=detect,
    inputs=["image"],
    outputs=["image"],
)
demo.launch()
