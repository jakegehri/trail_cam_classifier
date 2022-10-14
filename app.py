from fastai.vision.all import *
import gradio as gr

__all__ = ['learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

learn = load_learner('model.pkl')
categories = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['deer.jpg', 'elk.jpg', 'moose.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()