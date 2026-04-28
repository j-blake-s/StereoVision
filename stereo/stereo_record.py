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
import threading


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--frames', default="", help="Path to save frames")
    parser.add_argument('-e', '--events', default="", help="Path to save events")
    args = parser.parse_args()
    return args

# --- Dedicated Video Writer Thread ---
class VideoWriterThread(threading.Thread):
    def __init__(self, filename, fps, width, height):
        super().__init__()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)
        self.frame_queue = queue.Queue(maxsize=128) # Larger buffer to handle spikes
        self.running = True

    def run(self):
        while self.running or not self.frame_queue.empty():
            try:
                # Timeout allows the loop to check self.running occasionally
                frame = self.frame_queue.get(timeout=0.1)
                self.writer.write(frame)
            except queue.Empty:
                continue
        self.writer.release()
        print("Video Writer Thread: Finished saving.")

    def stop(self):
        self.running = False

class VimbaHandler:
    def __init__(self, writer_queue):
        self.writer_queue = writer_queue
        self.latest_frame = None

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            img = frame.as_numpy_ndarray().copy()
            # Direct push to writer thread
            if not self.writer_queue.full():
                self.writer_queue.put(img)
            
            # Keep a reference for the UI display
            self.latest_frame = img
            
        cam.queue_frame(frame)

def main():

    args = parse_args()
    frame_fn = args.frames if args.frames else "data/frames_recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".mp4"
    event_fn = args.events if args.events else "data/events_recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".raw"
    
    # HAL Device on live camera
    device = initiate_device("")
    
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

            target_fps = 25
            cam.get_feature_by_name('AcquisitionFrameRateEnable').set(True)
            cam.get_feature_by_name('AcquisitionFrameRate').set(target_fps)

            # exposure_time = cam.get_feature_by_name('ExposureTime')
            # exposure_time.set(20000.0) 
            
            # Set Gain if exposure isn't enough (e.g., 10.0 dB)
            gain = cam.get_feature_by_name('Gain')
            gain.set(10.0)

            # Get original dimensions for the video writer
            orig_w = int(cam.get_feature_by_name('Width').get())
            orig_h = int(cam.get_feature_by_name('Height').get())

            # Start the Writer Thread
            writer_thread = VideoWriterThread(frame_fn, target_fps, orig_w, orig_h)
            writer_thread.start()
            print(f'Streaming frames to {frame_fn}')

            vimba_handler = VimbaHandler(writer_thread.frame_queue)

            if device.get_i_events_stream():
                print(f'Streaming events to {event_fn}')
                cam.start_streaming(handler=vimba_handler, buffer_count=30)
                device.get_i_events_stream().log_raw_data(event_fn)

            with MTWindow(title="RECORDING - Dual View", width=ev_w * 2, height=ev_h,
                         mode=BaseWindow.RenderMode.BGR) as window:
                def keyboard_cb(key, scancode, action, mods):
                    if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                        window.set_close_flag()
                window.set_keyboard_callback(keyboard_cb)

                event_frame_gen = PeriodicFrameGenerationAlgorithm(
                    sensor_width=ev_w, sensor_height=ev_h, fps=target_fps, palette=ColorPalette.Dark)

                def on_cd_frame_cb(ts, cd_frame):
                    if vimba_handler.latest_frame is not None:
                        v_frame = cv2.cvtColor(vimba_handler.latest_frame, cv2.COLOR_GRAY2BGR)
                        v_frame = cv2.resize(v_frame, (ev_w, ev_h))
                    else:
                        v_frame = np.zeros((ev_h, ev_w, 3), dtype=np.uint8)
                    combined = np.hstack((v_frame, cd_frame))
                    window.show_async(combined)
                event_frame_gen.set_output_callback(on_cd_frame_cb)

                try:
                    for evs in mv_iterator:
                        EventLoop.poll_and_dispatch()
                        event_frame_gen.process_events(evs)
                        if window.should_close():
                            break
                finally:
                    print("Stopping and saving...")
                    device.get_i_events_stream().stop_log_raw_data()
                    cam.stop_streaming()
                    writer_thread.stop() # Tell writer to finish the queue and close
                    writer_thread.join() # Wait for it to finish writing

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()













