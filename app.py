import streamlit as st
from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
import io
from PIL import Image

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
prompt_text = st.text_input("Enter a prompt for the image:", "")

# Modify the prompt to exclude human body cloth
prompt = f"{prompt_text} without human body"

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

    # Resize to 224x224, then convert to grayscale and apply Gaussian blur
    resized_image = cv2.resize(image, (224, 224))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Binarize and equalize histogram
    _, binarized_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    equalized_image = cv2.equalizeHist(binarized_image)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    _, binarized_edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    equalized_edges = cv2.equalizeHist(binarized_edges)

    # Save processed images
    processed_edges_path = "processed_edges.png"
    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_edges_path, equalized_edges)
    cv2.imwrite(processed_image_path, equalized_image)

    # Display the generated image and add a download button for it
    st.image(generated_image, caption="Generated Image", use_column_width=True)
    with open(generated_image_path, "rb") as f:
        st.download_button(
            label="Download Generated Image",
            data=f,
            file_name=generated_image_path,
            mime="image/png",
        )

    # Display the preprocessed images side-by-side with download buttons
    col1, col2 = st.columns(2)  # Create two columns for the preprocessed images

    with col1:
        st.image(equalized_image, caption="Processed Image", use_column_width=True)
        with open(processed_image_path, "rb") as f:
            st.download_button(
                label="Download Processed Image",
                data=f,
                file_name=processed_image_path,
                mime="image/png",
            )

    with col2:
        st.image(equalized_edges, caption="Processed Edges", use_column_width=True)
        with open(processed_edges_path, "rb") as f:
            st.download_button(
                label="Download Processed Edges",
                data=f,
                file_name=processed_edges_path,
                mime="image/png",
            )
