import cv2
import queue
import time
from vmbpy import *
import argparse

# Queue for thread-safe frame passing
frame_queue = queue.Queue(maxsize=2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--filename', default="", help="filename")
    args = parser.parse_args()
    return args

class VimbaHandler:
    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # Copy frame to avoid buffer overwriting
            img = frame.as_numpy_ndarray().copy()
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(img)
        cam.queue_frame(frame)

def main():
    args = parse_args()
    filename = args.filename if args.filename else "recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".mp4"
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No Allied Vision cameras found.")
            return

        with cams[0] as cam:
            # Setup camera settings (Exposure, Gain, etc. can be set here)
            # Ensure the pixel format is set to a standard monochrome format
            cam.set_pixel_format(PixelFormat.Mono8)

            # Set Exposure Time in microseconds (e.g., 20,000us = 20ms)
            # Note: Max exposure is limited by your desired Frame Rate
            exposure_time = cam.get_feature_by_name('ExposureTime')
            exposure_time.set(20000.0) 
            
            # Set Gain if exposure isn't enough (e.g., 10.0 dB)
            gain = cam.get_feature_by_name('Gain')
            gain.set(15.0)

            # Get original dimensions for the video writer
            orig_w = int(cam.get_feature_by_name('Width').get())
            orig_h = int(cam.get_feature_by_name('Height').get())

            # 'mp4v' is widely compatible. Change filename as needed.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Important: isColor=False for Mono8 frames
            out = cv2.VideoWriter(filename, fourcc, 60.0, (orig_w, orig_h), isColor=False)

            try:
                # Start Asynchronous Streaming
                vimba_handler = VimbaHandler()
                cam.start_streaming(handler=vimba_handler, buffer_count=10)
                print("Streaming started. Press 'q' to stop.")


                while True:
                    if not frame_queue.empty():
                        raw_frame = frame_queue.get()

                        # Write the RAW full-resolution frame to disk
                        out.write(raw_frame)

                        # Resize ONLY for display to keep the script responsive
                        scale = 0.4
                        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
                        display_frame = cv2.resize(raw_frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow('Recording Feed (Preview)', display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                print("Cleaning up...")
                # Allow the queue to empty and the driver to breathe
                time.sleep(0.5) 
                try:
                    cam.stop_streaming()
                except Exception as e:
                    print(f"Ignored stop error: {e}")

                out.release()
                cv2.destroyAllWindows()
                print("Recording saved.")

if __name__ == "__main__":
    main()





