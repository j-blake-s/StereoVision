import cv2
from vmbpy import *
import queue

# A simple queue to pass frames from the camera thread to the main thread
frame_queue = queue.Queue(maxsize=2)

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
            
            try:
                # Start Asynchronous Streaming
                vimba_handler = VimbaHandler()
                cam.start_streaming(handler=vimba_handler, buffer_count=10)
                print("Streaming started. Press 'q' to stop.")

                scale_percent = 0.4
                while True:
                    if not frame_queue.empty():
                        img = frame_queue.get()

                        # Resize using the fastest interpolation
                        width = int(img.shape[1] * scale_percent)
                        height = int(img.shape[0] * scale_percent)
                        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

                        cv2.imshow('Live Allied Vision Stream', resized_img)

                    # WaitKey(1) is essential to let OpenCV handle the window events
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cam.stop_streaming()

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()