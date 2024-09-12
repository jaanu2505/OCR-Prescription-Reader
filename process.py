import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import json

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_data(image_path,json_path,save_dir):
    with open(json_path, "r") as f:
        data= json.load(f)

    img_metadata = data.get("_via_img_metadata", {})
    for img_key, img_value in tqdm(img_metadata.items()):
        filename = img_value.get("filename")
        if filename:
            print(f"Processing image: {filename}")
            # Process regions within the image
            regions = img_value.get("regions", [])
            for region in regions:
                region_attributes = region.get("region_attributes", {})
                composition = region_attributes.get("COMPOSITION", "Unknown composition")
                type_of_medicine = region_attributes.get("type", "Unknown type")
                print(f"  Found region: {region_attributes.get('name', 'Unknown name')}")
                print(f"    Composition: {composition}")
                print(f"    Type: {type_of_medicine}")
        else:
            print(f"Filename key not found in entry: {img_key}")

        break


if __name__ == "__main__":
    """ Dataset path """
    dataset_path = r"C:\Users\aujal\OneDrive\Desktop\python_OCR_application\dataset"
    dataset= glob(os.path.join(dataset_path, "*"))

    print("Dataset directories found:", dataset)

    for data in dataset:
        image_path = glob(os.path.join(data, "images", "image*"))[0]
        json_path = glob(os.path.join(data,"images", "*.json"))[0]
        # print(f"Images found in {data}: {image_path}")
        # print(f"json file found {data}: {json_path}")
        save_dir= f"data/{"main"}/"
        create_dir(f"{save_dir}image")
        create_dir(f"{save_dir}/mask")

        process_data(image_path,json_path,save_dir)