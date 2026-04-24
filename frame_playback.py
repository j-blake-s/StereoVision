import cv2
import time
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--filename', required=True, help="filename")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    video_path = args.filename  # Ensure this matches your recorded filename
    
    # 1. Initialize video capture
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return


    # 2. Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # print(f"Playing: {video_path}")
    # print(f"Resolution: {width}x{height} | FPS: {fps}")
    # print("Press 'q' to stop playback.")

    # 3. Playback Loop
    while cap.isOpened():
        ret, frame = cap.read()

        # If ret is False, we reached the end of the video
        if not ret:
            # print("End of video reached.")
            break

        # Optional: Resize for screen if the original is too big
        # Since this is a file, resizing won't cause the "Internal Fault" errors
        display_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

        cv2.imshow('MP4 Playback', display_frame)

        # 4. Timing Control
        # waitKey(ms) handles the frame rate. 
        # For a 30fps video, 1000/30 = ~33ms
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    # 5. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()