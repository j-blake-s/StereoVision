import os

def rename_frames(folder, fps):
    # Calculate nanoseconds per frame
    ns_per_frame = int(1e9 / fps)
    
    # Get all png files and sort them numerically
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    
    print(f"Renaming {len(files)} files in {folder}...")
    
    for i, filename in enumerate(files):
        # Calculate timestamp
        timestamp_ns = i * ns_per_frame
        
        # Format to 19 digits with leading zeros
        new_name = f"{timestamp_ns:019d}.png"

        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        
        os.rename(src, dst)

if __name__ == "__main__":
    # Do it for both camera folders
    for cam in ['dataset/cam0', 'dataset/cam1']:
        if os.path.exists(cam):
            rename_frames(cam, 25)
    print("Done!")