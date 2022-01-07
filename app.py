import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import os
from tempfile import NamedTemporaryFile


# Load model from TF-Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2') 
content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
style_file_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'
style_img_size = (512, 512) 
content_img_size = (512, 512) 


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
        tf.io.read_file(image_url),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    # img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def save_uploadedfile(uploadedfile):
    with open("./images/" + uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return "images/" + uploadedfile.name


st.set_page_config(layout="wide")
col1, col2 = st.columns(2)


placeholder = st.container()


with col1:
    st.markdown("## Content Image")
    image_file = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
    if(image_file):
        st.image(image_file, width=200)

with col2:
    st.markdown("## Style Image")
    style_file = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
    if(style_file):
        st.image(style_file, width=200)


with placeholder:
    if style_file and image_file:
        with st.spinner('Generating Stylised Image...'):
            st.markdown("## Generated Image")
            content_path = save_uploadedfile(image_file)
            style_path = save_uploadedfile(style_file)

            content_image = load_image(content_path, content_img_size)
            style_file = load_image(style_path, style_img_size)
            # style_file = tf.nn.avg_pool(style_file, ksize=[3, 3], strides=[1, 2], padding='SAME')
            outputs = hub_model(content_image, style_file)
            stylized_image = outputs[0]
            st.image(tensor_to_image(stylized_image))
        st.success('Done!')
