import streamlit as st
from diffusers import DiffusionPipeline
import torch
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
st.title("Generate your Own cloth")

# Input for the text prompt
prompt = st.text_input("Enter a prompt for the image:", "")

# Generate the image
if st.button("Generate Image"):
    pipe = load_pipeline()

    # Append "+mockup" to the end of the prompt
    full_prompt = f"{prompt} +mockup"

    with st.spinner("Generating image..."):
        # Generate the image
        generated_image = pipe(prompt=full_prompt).images[0]

    # Save the generated image to an in-memory buffer
    buf = io.BytesIO()
    generated_image.save(buf, format="PNG")
    buf.seek(0)

    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_column_width=True)
    
    # Add a download button for the generated image
    st.download_button(
        label="Download Generated Image",
        data=buf,
        file_name="generated_image.png",
        mime="image/png",
    )
