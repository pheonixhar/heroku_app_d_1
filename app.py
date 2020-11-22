
import gradio as gr
from fastai.vision.all import *
import numpy as np
import requests

inception_net = tf.keras.applications.InceptionV3() # load the model

path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'https://drive.google.com/file/d/1dT0AMkVpHZxCu5IG5TlTpksZ7ZyeSyWh/view?usp=sharing')

labels = ['fake', 'real']
def classify_image(img):
    img = PILImage.create(img).resize((224,224))

    pred,pred_idx,probs = learn_inf.predict(PILImage(img))
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

image = gr.inputs.Image(shape=(224, 224, 3))
label = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=classify_image, inputs=image, outputs=label, capture_session=True).launch()