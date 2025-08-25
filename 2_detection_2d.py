import os
import json
import argparse
import cv2
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run YOLO inference on a folder of images')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--model', type=str, default='data/ckpts/best.pt', help='Path to YOLO model')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--output-dir', type=str, default='data/inference_results', help='Output directory')
    return parser.parse_args()

def create_output_directory(input_folder, output_base_dir):
    # Extract the last part of the input folder path for the subfolder name
    subfolder_name = os.path.basename(input_folder)
    output_dir = os.path.join(output_base_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image"""
    img_copy = image.copy()
    for det in detections:
        bbox = det["bbox"]
        label = det["label"]
        score = det["score"]
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and score
        text = f"{label}: {score:.2f}"
        cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_copy

def main():
    args = parse_arguments()
    
    # Load the YOLO model
    model = YOLO(args.model)
    
    # Class names from the configuration
    class_names = [
        'Gun', 'Bullet', 'Magazin', 'Slingshot', 'Speargun tip', 'Shuriken', 
        'Dart pin', 'Electroshock weapon', 'LAGs product', 'Ax', 'Knife-A', 
        'Knife-F', 'Knife-B', 'Other Knife', 'Knife-D', 'Knife blade', 'Knife-Z', 
        'Multipurpose knife', 'Scissors-A', 'Scissors-B', 'Knuckle', 'Hammer', 
        'Prohibited tool-D', 'Drill', 'Prohibited tool-A', 'Monkey wrench', 
        'Pipe wrench', 'Prohibited tool-C', 'Prohibited tool-B', 'Vise plier', 
        'Shovel', 'Prohibited tool-E', 'Bolt cutter', 'Saw', 'Electric saw', 
        'Dumbbel', 'Ice skates', 'Baton', 'Handscuffs', 'Explosive weapon-A', 
        'LAGs product(Plastic-A)', 'LAGs product(Plastic-B)', 'LAGs product(Plastic-C)', 
        'LAGs product(Plastic-D)', 'LAGs product(Glass)', 'LAGs product(Paper)', 
        'LAGs product(Stainless)', 'LAGs product(Vinyl)', 'LAGs product(Aluminum)', 
        'LAGs product(Tube)', 'Firecracker', 'Torch', 'Solid fuel', 'Lighter', 
        'Nunchaku', 'Exploding golf balls', 'Knife-E', 'Green onion slicer', 
        'Hex key(over 10cm)', 'Kettlebell', 'Kubotan', 'Arrow tip', 'Billiard ball', 
        'Drill bit(over 6cm)', 'Buttstock', 'Card knife'
    ]
    
    # Create output directory
    output_dir = create_output_directory(args.input, args.output_dir)
    
    # Process each image in the input folder
    image_files = [f for f in os.listdir(args.input) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(args.input, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
        

        results = model(img, conf=args.conf_thres)[0]
        

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = class_names[cls_id]
            
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "label_id": cls_id,
                "score": conf
            })
        

        json_output = {
            "count": len(detections),
            "data": detections
        }

        output_json_path = os.path.join(output_dir, img_file.rsplit('.', 1)[0] + '.json')
        output_img_path = os.path.join(output_dir, img_file)
        
        # Save JSON output
        with open(output_json_path, 'w') as f:
            json.dump(json_output, f, indent=4)
    
        img_with_detections = draw_detections(img, detections)
        cv2.imwrite(output_img_path, img_with_detections)
        
        print(f"Processed {img_file} - Found {len(detections)} objects")

if __name__ == "__main__":
    main()