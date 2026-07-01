from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generate an image using DALL-E
response = client.images.generate(
    model="dall-e-3",
    prompt="A sleek, humanoid robot with a polished, metallic surface that reflects light in beautiful patterns. Its design is streamlined and elegant, resembling the contours of the human body but with futuristic features. The robot has articulated limbs, capable of fluid movements, allowing it to strike a pose reminiscent of classic human aesthetics.",
    size="1024x1024",
    quality="standard",
    n=1,
)

print(f"Image URL: {response.data[0].url}")
print(f"\nImage has been generated! Open the URL above to view it.")
