"""
This module provides functionality to extract
image embeddings using a specified
pretrained model from the torchvision library.
It includes functions to:
- Load and process images from a ZIP file.
- Apply model-specific preprocessing and transformations.
- Extract embeddings using various models.
- Save the resulting embeddings into a CSV file.
Modules required:
- argparse: For command-line argument parsing.
- os, csv, zipfile: For file handling (ZIP file extraction, CSV writing).
- inspect: For inspecting function signatures and models.
- torch, torchvision: For loading and
using pretrained models to extract embeddings.
- PIL, cv2: For image processing tasks
such as resizing, normalization, and conversion.
"""
import argparse
import csv
import logging
import zipfile
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.models as models
from torchvision import transforms
import inspect  # Added this import to fix the NameError

# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Define available models from torchvision
AVAILABLE_MODELS = {
    name: getattr(models, name)
    for name in dir(models)
    if callable(getattr(models, name)) and "weights" in inspect.signature(getattr(models, name)).parameters
}

MODEL_DEFAULTS = {
    "default": {"resize": (224, 224), "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])},
    "vgg11": {"resize": (224, 224)},  # Adjusted for your use case (vgg11)
}

for model, settings in MODEL_DEFAULTS.items():
    if "normalize" not in settings:
        settings["normalize"] = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def load_model(model_name, device, grayscale_native=False):
    """Loads a model and optionally adapts it for grayscale input."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    logging.info(f"Loading model: {model_name}")
    model = AVAILABLE_MODELS[model_name](weights="DEFAULT").to(device)
    
    if grayscale_native and hasattr(model, 'features'):  # For VGG-like models
        logging.info("Adapting model for native grayscale input")
        original_conv1 = model.features[0]  # VGG uses features[0] as first conv layer
        model.features[0] = torch.nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        with torch.no_grad():
            model.features[0].weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))
    
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, 'head'):
        model.head = torch.nn.Identity()
    
    model.eval()
    logging.info("Model loaded successfully")
    return model

def process_image(file, transform, device, transform_type="rgb", grayscale_native=False):
    """Loads and transforms an image from a file-like object."""
    logging.info(f"Applying transform type: {transform_type}")
    if transform_type in ["clahe", "edges"]:
        file.seek(0)
        arr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if transform_type == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        elif transform_type == "edges":
            image = cv2.Canny(image, threshold1=100, threshold2=200)
        image = Image.fromarray(image).convert("RGB")
    else:
        image = Image.open(file)
        if transform_type == "grayscale":
            image = image.convert("L")
            if not grayscale_native:
                image = image.convert("RGB")
        elif transform_type == "rgba_to_rgb":
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def main(zip_file, output_csv, model_name, apply_normalization=False, transform_type="rgb", ludwig_format=False, grayscale_native=False):
    """Main function to process images and extract embeddings."""
    logging.info(f"Starting to process ZIP file: {zip_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = load_model(model_name, device, grayscale_native)
    model_settings = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["default"])
    resize = model_settings["resize"]
    
    if apply_normalization:
        if grayscale_native and transform_type == "grayscale":
            normalize = ([0.5], [0.5])
        else:
            normalize = model_settings["normalize"]
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
    logging.info(f"Transformations set: resize={resize}, normalize={normalize if apply_normalization else 'None'}")
    
    channels = 1 if grayscale_native and transform_type == "grayscale" else 3
    dummy_input = torch.zeros(1, channels, *resize).to(device)
    with torch.no_grad():
        embedding = model(dummy_input).squeeze()
    embedding_size = 1 if embedding.dim() == 0 else embedding.shape[0]
    logging.info(f"Embedding size determined: {embedding_size}")
    
    logging.info(f"Opening CSV file: {output_csv}")
    with open(output_csv, mode="w", encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["sample_name"] + ([f"vector{i+1}" for i in range(embedding_size)] if not ludwig_format else ["embedding"])
        csv_writer.writerow(header)
        logging.info("CSV header written")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logging.info(f"Found {len(file_list)} files in ZIP")
            for file_name in file_list:
                logging.info(f"Processing image: {file_name}")
                try:
                    with zip_ref.open(file_name) as file:
                        input_tensor = process_image(file, transform, device, transform_type, grayscale_native)
                        with torch.no_grad():
                            embedding = model(input_tensor).squeeze().cpu().numpy()
                        if ludwig_format:
                            embedding_str = " ".join(map(str, embedding))
                            csv_writer.writerow([file_name, embedding_str])
                        else:
                            csv_writer.writerow([file_name] + embedding.tolist())
                        logging.info(f"Embedding written for {file_name}")
                except Exception as e:
                    logging.warning(f"Error processing {file_name}: {e}")
    
    logging.info("Finished processing all images")
    logging.info(f"CSV file created at: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings.")
    parser.add_argument('--zip_file', required=True, help="Path to ZIP file")
    parser.add_argument('--model_name', required=True, choices=AVAILABLE_MODELS.keys(), help="Model name")
    parser.add_argument('--normalize', action="store_true", help="Apply normalization")
    parser.add_argument('--transform_type', required=True, help="Image transformation type")
    parser.add_argument('--output_csv', required=True, help="Output CSV file")
    parser.add_argument('--ludwig_format', action="store_true", help="Use Ludwig format")
    parser.add_argument('--grayscale_native', action="store_true", help="Keep grayscale as 1-channel")
    
    args = parser.parse_args()
    main(args.zip_file, args.output_csv, args.model_name, args.normalize, args.transform_type, args.ludwig_format, args.grayscale_native)
