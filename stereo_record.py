import cv2
import queue
import time
import numpy as np
from vmbpy import *
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
import argparse



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--filename', default="", help="filename")
    args = parser.parse_args()
    return args


# --- Vimba Frame Handler ---
frame_queue = queue.Queue(maxsize=2)

class VimbaHandler:
    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            img = frame.as_numpy_ndarray().copy()
            if not frame_queue.full():
                frame_queue.put(img)
        cam.queue_frame(frame)



def main():

    args = parse_args()
    filename = args.filename if args.filename else "data/recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
    
    # HAL Device on live camera
    device = initiate_device("")

    # Start the recording
    if device.get_i_events_stream():
        print(f'Recording to {filename}.raw')
        device.get_i_events_stream().log_raw_data(f'{filename}.raw')
    
    
    # Initialize Event Camera & Writer
    mv_iterator = EventsIterator.from_device(device=device)
    ev_h, ev_w = mv_iterator.get_size()


    # Initialize Vimba System
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams: return
        
        with cams[0] as cam:
            # Camera Config
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
            vid_writer = cv2.VideoWriter(f"{filename}.mp4", fourcc, 60.0, (orig_w, orig_h), isColor=False)

            # 4. Setup Combined Display Window
            with MTWindow(title="RECORDING - Dual View", width=ev_w * 2, height=ev_h,
                         mode=BaseWindow.RenderMode.BGR) as window:
                def keyboard_cb(key, scancode, action, mods):
                    if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                        window.set_close_flag()
                window.set_keyboard_callback(keyboard_cb)

                event_frame_gen = PeriodicFrameGenerationAlgorithm(
                    sensor_width=ev_w, sensor_height=ev_h, fps=60, palette=ColorPalette.Dark)

                # Store latest frame for the tiled display
                current_frame_img = np.zeros((ev_h, ev_w, 3), dtype=np.uint8)

                def on_cd_frame_cb(ts, cd_frame):
                    combined_view = np.hstack((current_frame_img, cd_frame))
                    window.show_async(combined_view)
                event_frame_gen.set_output_callback(on_cd_frame_cb)

                # 5. Start Capture
                vimba_handler = VimbaHandler()
                cam.start_streaming(handler=vimba_handler, buffer_count=10)
                print(f"COMMENCING CAPTURE: Saving to {filename}.(raw/mp4)")

                try:
                    for evs in mv_iterator:
                        EventLoop.poll_and_dispatch()
                        
                        # Save & Process Events
                        event_frame_gen.process_events(evs)

                        # Save & Process Frames
                        if not frame_queue.empty():
                            raw_frame = frame_queue.get()
                            vid_writer.write(raw_frame) # Save full res
                            
                            # Update display image (resize to match event height)
                            # Convert to BGR for the MTWindow tiling
                            disp_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
                            current_frame_img = cv2.resize(disp_frame, (ev_w, ev_h))

                        if window.should_close():
                            device.get_i_events_stream().stop_log_raw_data()
                            break
                finally:
                    print("Stopping and saving...")
                    cam.stop_streaming()
                    vid_writer.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()













