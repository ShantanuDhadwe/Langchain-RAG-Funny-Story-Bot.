

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
import io
import matplotlib.pyplot as plt

load_dotenv()  

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=HF_TOKEN)


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" 
def generate_img(prompt):
    PROMPT = prompt
    NEGATIVE_PROMPT = "blurry, low quality"
    image = None
    try:
        image = client.text_to_image(
            PROMPT,
            model=MODEL_ID,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=9.0,
            num_inference_steps=30,
            width=768,
            height=768,  
        )
        print("✅ Image generated successfully!")
        return image
    except Exception as e:
        print("⚠️ Request failed:", e)
        if "BadRequestError" in str(e):
            print("\nPossible causes:")
            print("- Invalid model ID or model not available")
            print("- Parameters out of acceptable range")
            print("- Token might not have access to this model")
            print("- Prompt might contain unsafe content (safety filters are applied automatically)")
        elif "RateLimitError" in str(e):
            print("\nYou've hit the rate limit for this API. Please try again later.")
        elif "ServerError" in str(e):
            print("\nHugging Face server error. The service might be temporarily unavailable.")
        return None

def display_image(image):
    """
    Display the generated image using matplotlib or PIL's show method.
    
    Args:
        image: PIL Image object returned by generate_img function
    """
    if image is None:
        print("No image to display. Image generation may have failed.")
        return
    
    # Method 1: Using matplotlib (works in most environments including notebooks)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Hide the axes
    plt.title("Generated Image")
    plt.show()
    


# Example usage
if __name__ == "__main__":
    PROMPT = "A chaotic tea party with the Mad Hatter, March Hare, and the Dormouse (as a cushion) surrounded by teapots, unusual cup-and-saucer pairs, and a tree with a door in the background.These characters are from book Alice in Wonderland"
    
    # Generate the image
    generated_image = generate_img(PROMPT)
    
    # Display the image if generation was successful
    if generated_image:
        display_image(generated_image)
        

