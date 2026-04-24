import cv2
import argparse
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Dual Sensor Playback.')
    parser.add_argument('-e', '--event_file', required=True, help="Path to .raw or .dat event file")
    parser.add_argument('-f', '--frame_file', required=True, help="Path to .mp4 frame file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Event Playback
    mv_iterator = EventsIterator(input_path=args.event_file, delta_t=1000)
    ev_height, ev_width = mv_iterator.get_size()

    # LiveReplay ensures the playback speed matches real-time
    mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # Initialize Frame Playback
    cap = cv2.VideoCapture(args.frame_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    microseconds_per_frame = 2000000 / fps
    last_frame_ts = -1

    # We'll store the latest frame camera image here
    # Initialize with a black image matching event sensor size
    current_frame_img = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)

    # Setup Metavision Window & Generator
    with MTWindow(title="Events Playback", width=ev_width*2, height=ev_height,
                 mode=BaseWindow.RenderMode.BGR) as window:
        
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=ev_width, sensor_height=ev_height, fps=fps, palette=ColorPalette.Dark)
        def on_cd_frame_cb(ts, cd_frame): 
            combined_view = np.hstack((current_frame_img, cd_frame))
            window.show_async(combined_view)
        event_frame_gen.set_output_callback(on_cd_frame_cb)

        print(f"Playing back:\nEvents: {args.event_file}\nFrames: {args.frame_file}")


        # Integrated Playback Loop
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            # Get the current timestamp from the event stream
            # This is the "Master Clock" for your dataset
            current_ts = evs['t'][-1] if len(evs) > 0 else last_frame_ts

            # Pull the next frame from the MP4
            if current_ts - last_frame_ts >= microseconds_per_frame:
                ret, frame = cap.read()
                if ret:
                    # Convert to BGR if it's Mono and resize to match event height
                    if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    current_frame_img = cv2.resize(frame, (ev_width, ev_height))
                    # cv2.imshow("Allied Vision Playback", frame_view)
                    last_frame_ts = current_ts

            if window.should_close(): break

    cap.release()

if __name__ == "__main__":
    main()


















