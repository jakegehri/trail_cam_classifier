from fastai.vision.all import *
import gradio as gr
from PIL import Image

learn = load_learner('model1.pkl')
categories = ('deer', 'elk', 'moose')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False)