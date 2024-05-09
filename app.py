import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io

# Cache the diffusion pipeline for reuse
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

# App layout and image generation logic
st.title("Stable Diffusion Image Generator")

prompt = st.text_input("Enter a prompt for the image:", "man blue trousers")

if st.button("Generate Image"):
    pipe = load_pipeline()
    with st.spinner("Generating image..."):
        generated_image = pipe(prompt=prompt).images[0]
    
    # Display generated image and provide download option
    st.image(generated_image, caption="Generated Image", use_column_width=True)
    buffered = io.BytesIO()
    generated_image.save(buffered, format="PNG")
    st.download_button(
        label="Download Image",
        data=buffered,
        file_name="generated_image.png",
        mime="image/png",
    )
