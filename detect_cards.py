import numpy as np
import cv2
import Cards
import MagicCards as mc

import sys


def doImage(file):
  img = cv2.imread(file)              # image read from file
  output_img = Cards.findCards(img)   # image with found cards

  cv2.waitKey()

def doVideo(file):

  # Load training images
  refs = Cards.loadRefs("prepped")
  
  cap = cv2.VideoCapture(file)    # video loaded from file
  # print(cap.isOpened())

  # Debug: Skip to frame 1200
  for i in range(3000):
    retrieve, frame = cap.read()
  
  i = 2000
  # Go through frames
  while cap.isOpened():  
    i+=1
    retrieve, frame = cap.read()          # retrieval status and image
    output_frame = Cards.findCards(frame,refs) # image with found cards

    #print("Frame: ", i)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

def test(file):
  if file.endswith(".mp4"):
    # if video
    doVideo(file)
  else:
    # prbly an image
    doImage(file)


if __name__ == "__main__":
  sys.exit(test(sys.argv[1]))


