# pip install transformers accelerate bitsandbytes pillow torch
# Download https://huggingface.co/google/paligemma-3b-mix-224

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import requests
import sys
import torch

def setup_model_processor(quantize_bits=32):
    # Load the model and processor
    model_id = "google/google/paligemma-3b-mix-224"     # or paligemma-3b-pt-224

    if quantize_bits == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quantization_config
        ).eval()
    elif quantize_bits == 16:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"   # "cudo:0"
        ).eval()
    else:   # 32 bit floats
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor



def process_image(model, processor, img_path):
    if img_path.startswith("http"):
        image = Image.open(requests.get(img_path, stream=True).raw)
    else:
        image = Image.open(img_path)

    # Define the prompt (e.g., for captioning)
    prompt = "caption en"   # "detect" or "segment"

    # Preprocess and generate
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    # 5. Decode output
    output = processor.batch_decode(
        generation[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output


def process_path(filepath: str, img_glob: str = "*.jpg"):
    model, processor = setup_model_processor(quantize_bits=4)

    img_files = list(Path(filepath).rglob(img_glob, case_sensitive=False))
    print(len(img_files), f"image files in {filepath}")

    for img_path in img_files:
        output = process_image(model, processor, img_path)
        print(f"{img_path}\t{output}")


if __name__ == "__main__":
    argv = sys.argv
    img_path = argv[1] if len(argv) > 1 else "."
    process_path(img_path)
