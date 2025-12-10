import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import base64
import json
import re

# Load Env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# Path to the user uploaded image
image_path = r"C:/Users/sejee/.gemini/antigravity/brain/22f90022-b9aa-430e-860b-476b9ea1ad14/uploaded_image_1765373337699.png"

def analyze():
    print(f"Analyzing {image_path}...")
    try:
        if not os.path.exists(image_path):
            print("Image file not found!")
            return

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        GEMINI_IMAGE_PROMPT = """
Analyze this image and classify it as ONE of these 3 categories ONLY:

**real_image** - Authentic camera photo (phone/camera taken)
**ai_generated** - AI-created/synthesized image
**screenshot** - Screen capture/digital composite

LOOK FOR THESE CLUES:
- **Screenshot**: UI elements, perfect edges, compression blocks, browser chrome, low noise variance
- **AI Generated**: Anatomical errors (extra fingers, weird hands), symmetrical artifacts, unnatural lighting/shadows, blurry text/logos
- **Real Photo**: Natural noise/grain, lens distortion, organic lighting, camera sensor artifacts

OUTPUT EXACTLY:
{
  "decision": "real_image" | "ai_generated" | "screenshot",
  "confidence": 85,  // 0-100
  "evidence": "2-3 specific visual clues you saw"
}

NEVER say "uncertain" - pick your best guess with realistic confidence.
"""
        response = model.generate_content([
            GEMINI_IMAGE_PROMPT,
            {"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(image_bytes).decode()
            }}
        ])
        
        print("Raw Response:", response.text)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze()
