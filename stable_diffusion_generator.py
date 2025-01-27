# File: chainlit_diffusion_app.py

import torch
from diffusers import StableDiffusionPipeline
from chainlit import run, Message, user_input

# Load the Stable Diffusion model once when the app starts
def load_pipeline(model_name="stabilityai/stable-diffusion-2-1", device="cuda"):
    """
    Load the Stable Diffusion pipeline.
    """
    print(f"Loading model: {model_name} on {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    return pipeline.to(device)

# Initialize the pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = load_pipeline(device=device)


@run
async def main():
    """
    Main Chainlit app logic to generate images interactively.
    """
    # Greet the user
    await Message(content="Welcome to the Stable Diffusion Image Generator!").send()

    # Get the user input prompt
    prompt = await user_input("Enter a prompt to generate an image:")
    if not prompt:
        await Message(content="Prompt is required to generate an image!").send()
        return

    # Additional parameters
    steps = await user_input("Enter the number of steps (default: 50):", float, 50)
    guidance_scale = await user_input("Enter the guidance scale (default: 7.5):", float, 7.5)
    seed = await user_input("Enter a random seed for reproducibility (optional):", int, None)
    width = await user_input("Enter the image width (default: 512):", int, 512)
    height = await user_input("Enter the image height (default: 512):", int, 512)

    # Generate the image
    await Message(content=f"Generating an image for the prompt: **{prompt}**...").send()

    generator = torch.manual_seed(seed) if seed is not None else None
    image = pipeline(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        width=int(width),
        height=int(height),
    ).images[0]

    # Save the image temporarily
    output_path = "generated_image.png"
    image.save(output_path)

    # Display the image to the user
    await Message(content=f"Here is the generated image for your prompt: **{prompt}**").send()
    await Message(file=output_path).send()

    # Thank the user
    await Message(content="Thank you for using the Stable Diffusion Image Generator!").send()
