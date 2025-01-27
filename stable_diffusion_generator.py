# File: stable_diffusion_generator.py

import torch
from diffusers import StableDiffusionPipeline
import os

def load_pipeline(model_name="stabilityai/stable-diffusion-2-1", device="cuda"):
    """
    Load a pretrained Stable Diffusion pipeline from Hugging Face.
    """
    print(f"Loading model: {model_name} on {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)
    return pipeline

def generate_image(prompt, output_path="output.png", steps=50, guidance_scale=7.5, seed=None, width=512, height=512):
    """
    Generate an image using the Stable Diffusion pipeline.
    
    Args:
        prompt (str): Text prompt to guide image generation.
        output_path (str): Path to save the generated image.
        steps (int): Number of inference steps for image generation.
        guidance_scale (float): Guidance scale for classifier-free guidance.
        seed (int): Optional seed for reproducibility.
        width (int): Image width (default: 512px).
        height (int): Image height (default: 512px).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device=device)

    # Set random seed for reproducibility
    generator = torch.manual_seed(seed) if seed is not None else None

    print(f"Generating image for prompt: '{prompt}'...")
    image = pipeline(
        prompt, 
        num_inference_steps=steps, 
        guidance_scale=guidance_scale, 
        height=height, 
        width=width, 
        generator=generator
    ).images[0]
    
    # Save the generated image
    image.save(output_path)
    print(f"Image saved to: {output_path}")

def main():
    """
    Main function to run the Stable Diffusion image generator.
    """
    print("Welcome to the Stable Diffusion Image Generator!")
    prompt = input("Enter a prompt for the image (e.g., 'a serene mountain lake at sunrise'): ")
    output_path = input("Enter the output file name (default: 'output.png'): ").strip() or "output.png"
    steps = int(input("Enter the number of inference steps (default: 50): ") or 50)
    guidance_scale = float(input("Enter guidance scale (default: 7.5): ") or 7.5)
    seed = input("Enter a seed for reproducibility (optional): ")
    seed = int(seed) if seed.isdigit() else None
    width = int(input("Enter the image width (default: 512): ") or 512)
    height = int(input("Enter the image height (default: 512): ") or 512)

    # Generate the image
    generate_image(prompt, output_path, steps, guidance_scale, seed, width, height)

if __name__ == "__main__":
    main()
