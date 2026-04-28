ffmpeg -i %1 -f image2 dataset/cam0/%%05d.png
ffmpeg -i %2 -f image2 dataset/cam1/%%05d.png