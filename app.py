import streamlit as st
from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
from PIL import Image
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

# Function to process the generated image
def process_image(image):
    # Apply pre-processing (resize, grayscale, threshold, etc.)
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

    return equalized_image, equalized_edges

# Streamlit layout
st.title("Stable Diffusion and Image Processing")

# Input for the text prompt
prompt_text = st.text_input("Enter a prompt for the image:", "")

# Modify the prompt to exclude human body cloth
prompt = f"{prompt_text} without human body and without background"

# Generate the image and apply pre-processing
if st.button("Generate and Process Image"):
    pipe = load_pipeline()

    with st.spinner("Generating image..."):
        # Generate the image
        generated_image = pipe(prompt=prompt).images[0]

    # Convert PIL image to numpy array
    generated_np = np.array(generated_image)

    # Process the generated image
    processed_image, processed_edges = process_image(generated_np)

    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_column_width=True)

    # Display the processed images side-by-side
    col1, col2 = st.columns(2)  # Create two columns for the processed images

    with col1:
        st.image(processed_image, caption="Processed Image", use_column_width=True)

    with col2:
        st.image(processed_edges, caption="Processed Edges", use_column_width=True)

    # Download buttons for the generated and processed images
    with io.BytesIO() as output:
        # Save generated image to memory buffer
        generated_image.save(output, format='PNG')
        st.download_button(
            label="Download Generated Image",
            data=output.getvalue(),
            file_name="generated_image.png",
            mime="image/png",
        )

    with io.BytesIO() as output:
        # Save processed image to memory buffer
        Image.fromarray(processed_image).save(output, format='PNG')
        st.download_button(
            label="Download Processed Image",
            data=output.getvalue(),
            file_name="processed_image.png",
            mime="image/png",
        )

    with io.BytesIO() as output:
        # Save processed edges to memory buffer
        Image.fromarray(processed_edges).save(output, format='PNG')
        st.download_button(
            label="Download Processed Edges",
            data=output.getvalue(),
            file_name="processed_edges.png",
            mime="image/png",
        )
