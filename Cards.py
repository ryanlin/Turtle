import numpy as np
import os
import cv2
import MagicCards as mc

def loadRefs(path):
  refs = []
  for subdir, subdir_list, files in os.walk(path):
    for file in files:
      if file.endswith(".png"):
        img = cv2.imread(subdir+"/"+file, cv2.IMREAD_GRAYSCALE)
        name = os.path.splitext(file)[0]
        refs.append(mc.RefCard(name,img))
  return refs

def orientCard(image, pts, w, h):
  """Flattens an image of a card into a top-down 200x300 perspective.
  Returns the flattened, re-sized, grayed image.
  See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
  temp_rect = np.zeros((4,2), dtype = "float32")

  s = np.sum(pts, axis = 2)

  tl = pts[np.argmin(s)]
  br = pts[np.argmax(s)]

  diff = np.diff(pts, axis = -1)
  tr = pts[np.argmin(diff)]
  bl = pts[np.argmax(diff)]

  # Need to create an array listing points in order of
  # [top left, top right, bottom right, bottom left]
  # before doing the perspective transform

  if w <= 0.8*h: # If card is vertically oriented
    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl

  if w >= 1.2*h: # If card is horizontally oriented
    temp_rect[0] = bl
    temp_rect[1] = tl
    temp_rect[2] = tr
    temp_rect[3] = br

  # If the card is 'diamond' oriented, a different algorithm
  # has to be used to identify which point is top left, top right
  # bottom left, and bottom right.
  
  if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
    # If furthest left point is higher than furthest right point,
    # card is tilted to the left.
    if pts[1][0][1] <= pts[3][0][1]:
      # If card is titled to the left, approxPolyDP returns points
      # in this order: top right, top left, bottom left, bottom right
      temp_rect[0] = pts[1][0] # Top left
      temp_rect[1] = pts[0][0] # Top right
      temp_rect[2] = pts[3][0] # Bottom right
      temp_rect[3] = pts[2][0] # Bottom left

      # If furthest left point is lower than furthest right point,
      # card is tilted to the right
    if pts[1][0][1] > pts[3][0][1]:
      # If card is titled to the right, approxPolyDP returns points
      # in this order: top left, bottom left, bottom right, top right
      temp_rect[0] = pts[0][0] # Top left
      temp_rect[1] = pts[3][0] # Top right
      temp_rect[2] = pts[2][0] # Bottom right
      temp_rect[3] = pts[1][0] # Bottom left
  
  maxWidth = 200
  maxHeight = 300

  # Create destination array, calculate perspective transform matrix,
  # and warp card image
  dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
  M = cv2.getPerspectiveTransform(temp_rect,dst)
  warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  #warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

  return warp


def goodRatio(ratio, margin):
  # Card Aspect Ratios ( magic cards are 88x63)
  ratio1 = 63/88  # standard orientation
  ratio2 = 88/63  # sideways(?) orientation

  # For game1.mp4
  # ratio1 = 86/108
  # ratio2 = 108/86


  lo = 1 - margin
  hi = 1 + margin

  if ratio >= (ratio1*lo) and ratio <= (ratio1*hi) :
    # If Vertical Orientation (?)
    return True
  elif ratio >= (ratio2*lo) and ratio <= (ratio2*hi) :
    # If Horizontal Orientation (?)
    return True
  
  return False

def goodSize(w, h, w_avg, h_avg, margin):
  lo = 1 - margin
  hi = 1 + margin
  
  # If wide enough
  if w >= (w_avg*lo) and w <= (w_avg*hi) :
    # If tall enough
    if h >= (h_avg*lo) and h <= (h_avg*hi) :
      return True

  return False

def goodDims(w, h, margin):
  w_ref = 108   # reference width
  h_ref = 86    # reference height
  lo = 1 - margin
  hi = 1 + margin
  
  # If wide enough
  if w >= (w_ref*lo) and w <= (w_ref*hi) :
    return True
    # If tall enough
  elif h >= (h_ref*lo) and h <= (h_ref*hi) :
      return True

  return False

def drawCards(img, rects, maybe=None, nope=None):
  # print(type(img), "\t", type(rects))
  if not(img.any()):
    return 0

  # if img.dtype == 'uint8':
  #   img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


  if nope:
    # Debugging: Draw average Rect
    # drawAverages(img, nope, (0,0,255))
  
    for i, rect in enumerate(nope):
      x = rect[0]   # top right corner x
      y = rect[1]   # top left corner y
      w = rect[2]   # rectangle width
      h = rect[3]   # rectangle height
      x2 = x+w      # bottom right corner x
      y2 = y+h      # bottom left corner x

      cv2.rectangle(img, (x,y), (x2,y2), (0,0,255), 1)

  if maybe:
    # Debugging: Draw average Rect
    # drawAverages(img, maybe, (50,200,200))

    for i, rect in enumerate(maybe):
      x = rect[0]   # top right corner x
      y = rect[1]   # top left corner y
      w = rect[2]   # rectangle width
      h = rect[3]   # rectangle height
      x2 = x+w      # bottom right corner x
      y2 = y+h      # bottom left corner x

      cv2.rectangle(img, (x,y), (x2,y2), (50,200,200), 1)

  # Debugging: Draw average Rect
  # drawAverages(img, rects, (0,255,0))
  for i, rect in enumerate(rects):
    x = rect[0]   # top right corner x
    y = rect[1]   # top left corner y
    w = rect[2]   # rectangle width
    h = rect[3]   # rectangle height
    x2 = x+w      # bottom right corner x
    y2 = y+h      # bottom left corner x

    cv2.rectangle(img, (x,y), (x2,y2), (0,255,0), 2)

def drawLabels(img, cards, labels):
  print(labels)
  print("cards:", len(cards), "labels: ", len(labels))
  for i, card in enumerate(cards):
    print(i)
    cX = card[0] - 10
    cY = card[1] - 10
    cv2.putText(img, labels[i], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    
def preprocessImage(img):
  # Convert to Grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Blur to reduce noise

  if not(img.shape[0] <= 160):
    blurred = cv2.blur(gray, (3,3))

  # Threshold (too slow for us)
  #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
  #                              cv2.THRESH_BINARY, 11, 2)

  # Canny Edge Detection
  edge_thresh1 = 50
  edge_thresh2 = 80
  canny = cv2.Canny(blurred, edge_thresh1, edge_thresh2)
  #edges = thresh

  # Morphological Operations (e.g erode, dilate)
  kernel = np.ones((7,7), np.uint8)   # kernel for op
  edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations = 1)
  #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 1)
  
  return edges, canny

def processCard(img):
  # Convert to Grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #blurred = cv2.blur(gray, (5,5))
  # Canny Edge Detection
  edge_thresh1 = 50
  edge_thresh2 = 80
  canny = cv2.Canny(gray, edge_thresh1, edge_thresh2)

  # Morphological zOperations (e.g erode, dilate)
  kernel = np.ones((3,3), np.uint8)   # kernel for op
  edges = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel, iterations=1)
  #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 1)
  
  return edges, canny

def processCandidate(img):
  # Convert to Grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #blurred = cv2.blur(gray, (5,5))

  # Canny Edge Detection
  edge_thresh1 = 50
  edge_thresh2 = 80
  canny = cv2.Canny(gray, edge_thresh1, edge_thresh2)

  # Morphological Operations (e.g erode, dilate)
  kernel = np.ones((3,3), np.uint8)   # kernel for op
  edges = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel, iterations = 1)
  #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 1)
  
  return edges, canny



def detectCards(img, refs):
  # Preprocess (e.g gray, blur, etc)
  edges, canny = preprocessImage(img)

  # Find "Contours" (e.g shapes, structures)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  appx = []

  # Find contour "Approximations", gives the important "corners" of a contour
  for i, c in enumerate(contours):
    peri = cv2.arcLength(c, True)
    appx.append(cv2.approxPolyDP(c,0.04*peri, True))

  # Find "upright" rectangles in point set (each rect is stored as (x_pos, y_pos, width, height))
  rects = []
  approx = []
  for c in appx:
    if len(c) == 4:
      #print(c)
      rects.append(cv2.boundingRect(c))
      approx.append(c)

  if not rects:
    # if empty, return nothing
    return [], [], [], edges

  # Rectangle stats
  rects = np.array(rects)
  cutoff = int (len(rects) * .8) # cutoff index for top (1-TOP)%
  widths = sorted( rects[:,2])[cutoff:]
  heights = sorted( rects[:,3])[cutoff:]

  # Get average rectangle stats
  w_avg = int( np.mean(widths) )
  h_avg = int( np.mean(heights) )

  # Return Variables
  cards = []
  maybe = []
  nope = []
  labels = []

  for i,rect in enumerate(rects):
    w = rect[2]
    h = rect[3]

    ratio = rect[2]/rect[3]

    if goodRatio(ratio, 0.20):
      if goodDims(w,h,0.1):    # rightside up?
        #print("w: ", w, "\th:", h)
        cards.append(rect)
        oriented = orientCard(img, approx[i], w, h)
        label = identifyCard(canny, rect, refs, oriented)
        labels.append(label)
        #cv2.imshow("oriented", oriented)
        #cv2.waitKey(0)
      elif goodDims(h,w,0.1):  # sideways?
        #print("w: ", w, "\th:", h)
        cards.append(rect)
        oriented = orientCard(img, approx[i], w, h)
        label = identifyCard(canny, rect, refs, oriented)
        labels.append(label)
      else:
        maybe.append(rect)
    else:
      nope.append(rect)
  
  print("\n\n")
  print("cards:", len(cards), "labels: ", len(labels))
  
  return cards, maybe, nope, edges, labels


def identifyCard(img, card, refs, oriented):
  print("\n")
  # print("approx:", approx)
  # x = card[0]
  # y = card[1]
  # w = card[2]
  # h = card[3]
  # x2 = x+w
  # y2 = y+h

  # pic = img[y:y2, x:x2]

  diff = 60000
  card_diff = 8000
  index = 0

  o_canny, _ = processCandidate(oriented)

  # Try segmenting blobs rather than edges

  # o_seg = cv2.
  #cv2.imshow("o_canny", o_canny)
  #cv2.waitKey(0)
  # print("diff start", ": ", diff)

  for i, ref in enumerate(refs):

    # Make flipped copy
    flipped_img = cv2.flip( cv2.flip(o_canny, 0), 1)

    # Crop both
    x2 = o_canny.shape[1]
    y2 = int(o_canny.shape[0]*.5)

    cropped_img = o_canny[0:y2, 0:x2]
    cropped_flip = flipped_img[0:y2, 0:x2]

    # Resize both
    resized_img = cv2.resize(cropped_img, (ref.image.shape[1],ref.image.shape[0]))
    resized_flip = cv2.resize(cropped_flip, (ref.image.shape[1],ref.image.shape[0]))
    
    # Threshold both
    _, thresh_img = cv2.threshold(resized_img,  100, 256, cv2.THRESH_BINARY)
    _, thresh_flip = cv2.threshold(resized_flip,  100, 256, cv2.THRESH_BINARY)

    # More Processes on both
    kernel = np.ones((2,2), np.uint8)   # kernel for op
    did_img = cv2.morphologyEx(thresh_img, cv2.MORPH_DILATE, kernel, iterations = 1)
    did_flip = cv2.morphologyEx(thresh_flip, cv2.MORPH_DILATE, kernel, iterations = 1)


    #print("img: ", resized_img.shape[0:2])
    #print("ref: ", ref.image.shape[0:2])
  
    flip = False
    # Subtract both
    curr_diff1 = cv2.absdiff(did_img, ref.image)
    curr_diff2 = cv2.absdiff(did_flip, ref.image)
    card_diff1 = int(np.sum(curr_diff1)/255)
    card_diff2 = int(np.sum(curr_diff2)/255)
    card_diff = np.min([card_diff1, card_diff2])
    if card_diff == card_diff2:
     flip = True

    print(ref.name, ":\tcard_diff1: ", card_diff1, " card_diff2: ", card_diff2)

    if card_diff < diff:
      diff = card_diff
      index = i
    
  if diff <= 3000:
    print("Diff: ", diff, "\tLabel: not a class")
    return "no_class"

  label = refs[index].name
  print("Diff: ", diff, "\tLabel: ", label)
  if flip:
    cv2.imshow("image", did_flip)
  else:
    cv2.imshow("image", did_img)

  return label

def findCards(img, refs):

  cards, maybe, nope, mask, labels = detectCards(img, refs)

  drawCards(img, cards)
  drawLabels(img, cards, labels)
  mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  drawCards(mask, cards, maybe, nope)

  # Debug: show image and mask
  cv2.imshow("img", img)
  cv2.imshow("mask", mask)

  return img