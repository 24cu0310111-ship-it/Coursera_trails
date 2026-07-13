import os
import sys

sys.path = [path for path in sys.path if os.path.abspath(path) != os.path.dirname(__file__)]

import gradio as gr

def process_text(number, text):
    return f"You entered number={number} and text={text}"

demo=gr.Interface(
    fn=process_text, 
    inputs=[
        gr.Textbox(label="Enter a number"),
        gr.Textbox(label="Enter some  text")
            ], 
    outputs=gr.Textbox(label="Output")
    )
def count_files(files):
    return f"Number of files uploaded: {len(files)}"

upload=gr.Interface(
    fn=count_files,
    inputs=gr.File(file_count="multiple",
        type="filepath",
        label="Upload or Drag Files Here"),
    outputs=gr.Textbox(label="Number of Files uploaded")
)

app = gr.TabbedInterface([demo, upload], ["Text", "Upload"])

app.launch()
