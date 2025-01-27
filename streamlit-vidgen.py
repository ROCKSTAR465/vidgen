# File: streamlit_diffusion_app.py

import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image

# Load the Stable Diffusion model once when the app starts
@st.cache_resource
def load_pipeline(model_name="stabilityai/stable-diffusion-2-1", device="cuda"):
    """
    Load the Stable Diffusion pipeline.
    """
    st.write(f"Loading model: {model_name} on {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    return pipeline.to(device)

# Initialize the pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = load_pipeline(device=device)

# Streamlit app layout
st.title("Stable Diffusion Image Generator")
st.write("Welcome to the Stable Diffusion Image Generator!")

# Get the user input prompt
prompt = st.text_input("Enter a prompt to generate an image:")
if not prompt:
    st.warning("Prompt is required to generate an image!")
else:
    # Additional parameters
    steps = st.number_input("Enter the number of steps (default: 50):", min_value=1, max_value=200, value=50)
    guidance_scale = st.number_input("Enter the guidance scale (default: 7.5):", min_value=0.0, value=7.5)
    seed = st.number_input("Enter a random seed for reproducibility (optional):", value=None)
    width = st.number_input("Enter the image width (default: 512):", min_value=64, max_value=1024, value=512)
    height = st.number_input("Enter the image height (default: 512):", min_value=64, max_value=1024, value=512)

    # Generate the image button
    if st.button("Generate Image"):
        st.write(f"Generating an image for the prompt: **{prompt}**...")

        generator = torch.manual_seed(seed) if seed is not None else None
        image = pipeline(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            width=int(width),
            height=int(height),
        ).images[0]

        # Display the image to the user
        st.image(image, caption=f"Generated image for prompt: **{prompt}**", use_column_width=True)

        # Thank the user
        st.success("Thank you for using the Stable Diffusion Image Generator!")
