import cv2
import os
import argparse

def video_to_frames(input_video, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_FPS, 10)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames from {input_video} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to frames")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the frames")

    args = parser.parse_args()
    video_to_frames(args.input, args.output)
    