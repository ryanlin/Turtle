import MagicCards as mc
import Cards
import os
import cv2
import numpy as np

def loadImages(path):
  refs = []
  for subdir, subdir_list, files in os.walk(path):
    for file in files:
      if file.endswith(".PNG"):
        img = cv2.imread(subdir+"/"+file, cv2.IMREAD_COLOR)
        name = os.path.splitext(file)[0]
        refs.append(mc.RefCard(name,img))
  return refs

def prepImages(images):
  for image in images:
    print(image.name)

    # Crop
    x2 = image.image.shape[1]
    y2 = int(image.image.shape[0]*.5)

    print("shape[0]: ", image.image.shape[0], "shape[1]: ", image.image.shape[1])
    print("x2: ", x2, "y2: ", y2)
    cropped_img = image.image[0:y2, 0:x2]

    image.ref, _ = Cards.processCard(cropped_img)
    cv2.imshow(image.name, image.ref)
    cv2.waitKey()

def savePrepped(images):
  dir_name = "prepped"
  if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
  for image in images:
    file_name = image.name+".png"
    save_name = dir_name+"/"+file_name
    print(save_name)
    if os.path.isfile(save_name):
      os.remove(save_name)
    cv2.imwrite(save_name, image.ref)

images = loadImages("refs")
prepImages(images)
savePrepped(images)
