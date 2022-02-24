import cv2
import os
# Import utilites
import modules.menu as m
import modules.capture as c
import modules.functions as f
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model = 'detect1_metadata.tflite'
savePath = '.'
def run(model: str, loadPath: str, savePath: str, camera_id: int, width: int, height: int, num_threads: int) -> None:
    gestureQueue = f.GestureQueue() 
    mymenu = m.MenuGesti(gestureQueue, loadPath, savePath)
    capture = c.Capture(mymenu, model, gestureQueue, camera_id, width, height, num_threads).start()
    mymenu.start()
    capture.stop()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default=os.path.join('SSD_Network','detect1_metadata.tflite') )
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=1280)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=720)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--savePath',
      help='Save Path for Images.',
      required=False,
      default='.')
  parser.add_argument(
      '--loadPath',
      help='Load Path for Images.',
      required=False,
      default='Images/')
  args = parser.parse_args()
  run(args.model, args.loadPath, args.savePath, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads))


if __name__ == "__main__" :
    main()