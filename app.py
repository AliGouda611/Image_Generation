import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import cv2
import numpy as np
import io

# Function to load the Stable Diffusion pipeline
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda")
    return pipe

# Streamlit layout
st.title("Stable Diffusion and Image Processing")

# Input for the text prompt
prompt = st.text_input("Enter a prompt for the image:", "man blue trousers")

# Generate the image and apply pre-processing
if st.button("Generate and Process Image"):
    pipe = load_pipeline()

    with st.spinner("Generating image..."):
        # Generate the image
        generated_image = pipe(prompt=prompt).images[0]

    # Save the generated image to a path
    generated_image_path = "generated_image.png"
    generated_image.save(generated_image_path)

    # Apply pre-processing (resize, grayscale, threshold, etc.)
    image = cv2.imread(generated_image_path)

    # Resize image to 224x224
    desired_width, desired_height = 224, 224
    resized_image = cv2.resize(image, (desired_width, desired_height))

    # Convert to grayscale and apply Gaussian blur
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Binarize with a threshold and equalize histogram
    _, binarized_image = cv2.threshold(blurred_image, 0.5, 1, cv2.THRESH_BINARY)
    equalized_image = cv2.equalizeHist(binarized_image)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Binarize and equalize histogram for edges
    _, binarized_edges = cv2.threshold(edges, 0, 1, cv2.THRESH_BINARY)
    equalized_edges = cv2.equalizeHist(binarized_edges)

    # Save the processed images
    processed_edges_path = "processed_edges.png"
    cv2.imwrite(processed_edges_path, equalized_edges)

    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_image_path, equalized_image)

    # Show the original and processed images
    st.image(generated_image, caption="Generated Image", use_column_width=True)
    st.image(equalized_image, caption="Processed Image", use_column_width=True)

    # Allow download of processed images
    with open(processed_edges_path, "rb") as f:
        st.download_button(
            label="Download Processed Edges",
            data=f,
            file_name=processed_edges_path,
            mime="image/png",
        )

    with open(processed_image_path, "rb") as f:
        st.download_button(
            label="Download Processed Image",
            data=f,
            file_name=processed_image_path,
            mime="image/png",
        )
