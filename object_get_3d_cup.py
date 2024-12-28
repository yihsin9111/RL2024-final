import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
from ultralytics import YOLO
import argparse
import os

def get_bounding_box(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    return (x_min, y_min), (x_max, y_max)

def calculate_distribution(array, scale=0.01):
    min_val = 0
    max_val = 1
    bins = np.arange(min_val, max_val + scale, scale)
    histogram, bin_edges = np.histogram(array, bins=bins)
    distribution = histogram
    return distribution, bin_edges

def detect_cup_with_yolo(image_path, depth_image, x_scale, y_scale):
    model = YOLO('yolov8s.pt') 
    results = model(image_path) 

    cup_box = None
    for box in results[0].boxes.data.cpu().numpy():
        x_min, y_min, x_max, y_max, confidence, class_id = box
        if class_id == 41:
            cup_box = (x_min, y_min, x_max, y_max)
            break

    if cup_box is None:
        print("No cup detected")
        return None, None, None

    x_min, y_min, x_max, y_max = cup_box
    depth_scale = []
    for x in range(int(x_min), int(x_max)):
        for y in range(int(y_min), int(y_max)):
            depth_scale.append(depth_image[int(y), int(x)] / 255)

    depth_distribute, bin_edges = calculate_distribution(depth_scale)

    obj_depth_min, obj_depth_max, boundary, depth_count = 0, 0, 0, 0
    find_start = False
    for i in range(len(depth_distribute) - 1, 0, -1):
        if not find_start and depth_distribute[i] >= 100:
            find_start = True
            obj_depth_min = 1 - bin_edges[i]
            boundary = depth_distribute[i]
            continue
        elif find_start and depth_distribute[i] < boundary and depth_count >= 5:
            obj_depth_max = 1 - bin_edges[i + 1]
            break
        elif find_start:
            depth_count += 1

    # print("Cup depth range:", obj_depth_min, obj_depth_max)

    box_coordinates_3d = {
        "x_min": x_min * x_scale,
        "y_min": y_min * y_scale,
        "z_min": obj_depth_min,
        "x_max": x_max * x_scale,
        "y_max": y_max * y_scale,
        "z_max": obj_depth_max,
    }

    return box_coordinates_3d, (x_min, y_min, x_max, y_max), (obj_depth_min, obj_depth_max)

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
IMAGE_FOLDER="./obj_detect/images"
def get_3d(image_path):
    image = Image.open(image_path)
    base_path = './obj_detect/output_images'
    output_original_image_path = f'{base_path}/original_image.jpg'
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_original_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    depth = pipe(image)["depth"]

    # save the depth image
    depth_image_path = f'{base_path}/depth_image.jpg'
    # depth.save(depth_image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(depth, cmap='gray')
    plt.axis('off')
    plt.savefig(depth_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()


    model = YOLO('yolov8s.pt')

    # Load image
    image = cv2.imread(image_path)
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_path = f'{base_path}/image_only_table.jpg'
    # plt figure size same as the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image)
    plt.axis('off') 
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    rgb_image_path = f'{base_path}/image_only_table.jpg'
    depth_image_path = f'{base_path}/depth_image.jpg'

    rgb_image = cv2.imread(rgb_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.imread(output_original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    print(f"depth_image.shape: {depth_image.shape}")

    # x and y scaling
    if original_image.shape[1] > original_image.shape[0]:
        x_scale = 1.0
        y_scale = original_image.shape[0]/original_image.shape[1]
    else:
        y_scale = 1.0
        x_scale = original_image.shape[1]/original_image.shape[0]
    
    mp_hands = mp.solutions.hands
    hand_centers = []
    displacements = []
    hand_skelotons_x = []
    hand_skelotons_y = []
    hand_skelotons_z = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.01) as hands:
        results_hands = hands.process(original_image)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # align hand depth
                x_min = min([lm.x for lm in hand_landmarks.landmark]) *  original_image.shape[1]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) *  original_image.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) *  original_image.shape[0]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) *  original_image.shape[0]

                hand_depth_scale = []
                for x in range(int(x_min), int(x_max)):
                    for y in range(int(y_min), int(y_max)):
                        hand_depth_scale.append(depth_image[y,x]/255)

                hand_depth_distribute, hand_bin_edges = calculate_distribution(hand_depth_scale)
                print("distribution: ", hand_depth_distribute)
                # print("bin edges: ", hand_bin_edges)
                plt.figure(figsize=(10, 6))
                plt.bar(hand_bin_edges[:-1], hand_depth_distribute, width=np.diff(hand_bin_edges), edgecolor='black', align='edge')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title('Distribution Histogram')
                plt.savefig("distribution_hand")
                plt.close()

                find_start = False
                obj_depth_min = 0
                obj_depth_max = 0
                boundary = 0
                depth_count = 0
                for i in range(hand_depth_distribute.shape[0]-1, 0, -1):
                    if not find_start and hand_depth_distribute[i] >= 50:
                        find_start = True
                        obj_depth_min = 1-hand_bin_edges[i]
                        boundary = hand_depth_distribute[i]
                        continue
                    elif find_start and hand_depth_distribute[i] < boundary and depth_count >= 5:
                        obj_depth_max = 1-hand_bin_edges[i+1] 
                        break
                    elif find_start:
                        depth_count += 1

                print("hand depth: ", obj_depth_min, obj_depth_max)
                
                z_min = min([lm.z for lm in hand_landmarks.landmark])
                z_max = max([lm.z for lm in hand_landmarks.landmark])

                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                hand_center_x = int(np.mean(x_list) * original_image.shape[1])
                hand_center_y = int(np.mean(y_list) * original_image.shape[0])
                hand_centers.append((hand_center_x, hand_center_y))
                x_list = [x_list[i] * x_scale for i in range(len(x_list))]
                y_list = [1-y_list[i] * y_scale for i in range(len(y_list))]
                z_list = [(obj_depth_min + (lm.z - z_min)/(z_max - z_min)*(obj_depth_max - obj_depth_min)) for lm in hand_landmarks.landmark]
                hand_skelotons_x.append(x_list)
                hand_skelotons_y.append(y_list)
                hand_skelotons_z.append(z_list)
    print(f"hand_centers: {hand_centers}")

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    box_coordinates_3d, cup_box_2d, depth_range = detect_cup_with_yolo(
        f'{base_path}/original_image.jpg', depth_image, x_scale, y_scale
    )

    if cup_box_2d is None:
        print("No cup detected")
        cup_data = None
    else:
        # Calculate the center of the 3D bounding box
        x_min, y_min, x_max, y_max = cup_box_2d
        z_min, z_max = depth_range

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        object_center = (x_center * x_scale, y_center * y_scale, z_center)

        # Calculate the eight 3D corners of the box
        corners = [
            (x_min * x_scale, y_min * y_scale, z_min),
            (x_max * x_scale, y_min * y_scale, z_min),
            (x_min * x_scale, y_max * y_scale, z_min),
            (x_max * x_scale, y_max * y_scale, z_min),
            (x_min * x_scale, y_min * y_scale, z_max),
            (x_max * x_scale, y_min * y_scale, z_max),
            (x_min * x_scale, y_max * y_scale, z_max),
            (x_max * x_scale, y_max * y_scale, z_max),
        ]

        cup_data = {
            "object_center": object_center,
            "corners": corners,
        }

    # Step 4: Compute 3D displacements (including depth calculation for hands and the cup)
    
    object_depth = None
    object_center = (x_center / x_scale, y_center / y_scale, z_center)
    if object_center:
        object_depth = depth_image[int(object_center[1]), int(object_center[0])]

    if object_center and hand_centers:
        for index, hand_center in enumerate(hand_centers):
            hand_depth = depth_image[hand_center[1], hand_center[0]]
            dx = object_center[0] - hand_center[0]
            dy = object_center[1] - hand_center[1]
            dz = object_depth - hand_depth
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            # hand_skelotons_z = [depth_image[hand_skelotons_y[index][i], hand_skelotons_x[index][i]] for i in range(len(hand_skelotons_x[index]))]
            hand_skelotons = np.array([hand_skelotons_x[index], hand_skelotons_z[index], hand_skelotons_y[index]]).T.tolist()
            displacements.append({
                'hand_center': hand_center,
                'hand_depth': int(hand_depth),  # Convert depth to an integer
                'dx': dx,
                'dy': dy,
                'dz': dz,
                '3D_distance': distance_3d,
                'hand_skelotons': hand_skelotons
            })
    elif hand_centers:
        for index, hand_center in enumerate(hand_centers):
            hand_depth = depth_image[hand_center[1], hand_center[0]]
            # hand_skelotons_z = [depth_image[hand_skelotons_y[index][i], hand_skelotons_x[index][i]] for i in range(len(hand_skelotons_x[index]))]
            hand_skelotons = np.array([hand_skelotons_x[index], hand_skelotons_z[index], hand_skelotons_y[index]]).T.tolist()
            displacements.append({
                'hand_center': hand_center,
                'hand_depth': int(hand_depth),  # Convert depth to an integer
                'dx': None,
                'dy': None,
                'dz': None,
                '3D_distance': None,
                'hand_skelotons': hand_skelotons
            })

    # Update the JSON data
    output_json_path = './obj_detect/3d_displacement_result.json'
    import json
    # Ensure all elements are JSON serializable
    data = {
        "image_path": image_path,
        "object": cup_data,
        "displacements": displacements
    }

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    try:
        with open(output_json_path, 'r') as f:
            if f.read(1):  # Check if the file has content
                f.seek(0)
                all_data = json.load(f)
            else:
                all_data = []
    except FileNotFoundError:
        all_data = []
    for i, d in enumerate(all_data):
        if d['image_path'] == image_path:
            all_data.pop(i)
            break
    all_data.append(json.loads(json.dumps(data, default=make_serializable)))

    with open(output_json_path, 'w') as f:
        json.dump(all_data, f, indent=4)

for image in os.listdir(IMAGE_FOLDER):
    get_3d(f"{IMAGE_FOLDER}/{image}")
    print(f"Done with {image}")
