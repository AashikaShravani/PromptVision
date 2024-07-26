import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from retry import retry
import os

# Set PYTORCH_CUDA_ALLOC_CONF to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Hey Pathaaaaaan !")
ctk.set_appearance_mode("dark")

# Change the background color to black
app.configure(bg='black')

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

@retry(tries=5, delay=2)
def load_pipeline():
    return StableDiffusionPipeline.from_pretrained(modelid)

pipe = load_pipeline()
pipe.to(device, torch.float16)
pipe.unet.to(device, torch.float16)
pipe.vae.to(device, torch.float16)
pipe.text_encoder.to(device, torch.float16)

def generate():
    # Clear any existing CUDA cache
    torch.cuda.empty_cache()

    # Generate image within torch.no_grad() to save memory
    with torch.no_grad():
        with autocast(device):
            result = pipe(prompt.get(), height=256, width=256, guidance_scale=8.5)
            image = result.images[0]

    # Move components back to CPU to free up memory
    pipe.unet.to('cpu')
    pipe.vae.to('cpu')
    pipe.text_encoder.to('cpu')

    # Save and display the image
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

    # Move components back to GPU for next generation
    pipe.unet.to(device)
    pipe.vae.to(device)
    pipe.text_encoder.to(device)

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="orange", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

# Set AI-based background image if available
background_image_path = "ai_background.png"
if os.path.exists(background_image_path):
    background_image = ImageTk.PhotoImage(file=background_image_path)
    background_label = tk.Label(app, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
else:
    print(f"Background image '{background_image_path}' not found. Skipping background image setup.")

app.mainloop()