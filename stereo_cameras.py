import cv2
import queue
from vmbpy import *
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
import numpy as np


# --- Allied Vision Handler ---
frame_queue = queue.Queue(maxsize=2)

class VimbaHandler:
    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # Copy the frame so Vimba can reuse the buffer immediately
            img = frame.as_numpy_ndarray().copy()
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(img)
        cam.queue_frame(frame)

def get_vimba_cam(vmb):
        # Discover cameras connected to the system
        cams = vmb.get_all_cameras()
        if not cams:
            print("No Allied Vision cameras found.")
            return

        # Access the first available camera
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

            return cam

def main():
    # 1. Initialize Event Camera
    mv_iterator = EventsIterator(input_path="", delta_t=1000)
    ev_h, ev_w = mv_iterator.get_size()
    current_frame_img = np.zeros((ev_h, ev_w, 3), dtype=np.uint8)


    with VmbSystem.get_instance() as vmb:
        with get_vimba_cam(vmb) as cam:
            with MTWindow(title="Dual Sensor View", width=ev_w * 2, height=ev_h,
                         mode=BaseWindow.RenderMode.BGR) as window:
                def keyboard_cb(key, scancode, action, mods):
                    if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                        window.set_close_flag()
                window.set_keyboard_callback(keyboard_cb)

                # Event Generator
                event_frame_gen = PeriodicFrameGenerationAlgorithm(
                    sensor_width=ev_w, sensor_height=ev_h, fps=60, palette=ColorPalette.Dark)
                def on_cd_frame_cb(ts, cd_frame): 
                    combined_view = np.hstack((current_frame_img, cd_frame))
                    window.show_async(combined_view)
                event_frame_gen.set_output_callback(on_cd_frame_cb)

                # Frame Generator
                vimba_handler = VimbaHandler()
                cam.start_streaming(handler=vimba_handler, buffer_count=10)
                
                print("Streaming both sensors. Press 'q' to quit.")

                # Combined Processing Loop
                for evs in mv_iterator:
                    EventLoop.poll_and_dispatch()
                    
                    # Process Event Data
                    event_frame_gen.process_events(evs)

                    # Process Frame Data (if available)
                    if not frame_queue.empty():
                        frame = np.squeeze(frame_queue.get())
                        if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        current_frame_img = cv2.resize(frame, (ev_w, ev_h))

                    if window.should_close(): break
                
                cam.stop_streaming()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()