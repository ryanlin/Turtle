import numpy as np
from scipy import stats
import cv2

# Debug: Check CV version
# print(cv2.__version__) # 4.1.1

global TOP
TOP = 0.9

"""


"""
def thresh_callback(thresh):
  return 0

def goodSize(w, h, w_avg, h_avg, margin):
  lo = 1 - margin
  hi = 1 + margin
  
  # If wide enough
  if w >= (w_avg*lo) and w <= (w_avg*hi) :
    return True
    # If tall enough
  elif h >= (h_avg*lo) and h <= (h_avg*hi) :
      return True

  return False

def goodRatio(ratio, margin):
  # Card Aspect Ratios ( magic cards are 88x63)
  ratio1 = 63/88  # standard orientation
  ratio2 = 88/63  # sideways(?) orientation

  lo = 1 - margin
  hi = 1 + margin

  if ratio >= (ratio1*lo) and ratio <= (ratio1*hi) :
    # If Vertical Orientation (?)
    return True
  elif ratio >= (ratio2*lo) and ratio <= (ratio2*hi) :
    # If Horizontal Orientation (?)
    return True
  
  return False

def drawAverages(img, rects, color=(100, 200, 200)):
  if not(img.any or rects):
    return 0

  rects = np.array(rects)
  cutoff = int (len(rects) * TOP) # cutoff index for top (1-TOP)%

  print("rects: ", len(rects), "\tcutoff: ", cutoff)

  if len(rects) == 0 :
    return 0

  widths = sorted( rects[:,2] )[cutoff:]
  heights = sorted( rects[:,3])[cutoff:]

  # Get average rectangle stats
  w_avg = int( np.median(widths) )
  h_avg = int( np.median(heights) )

  # Draw average rectangle
  cv2.rectangle(img, (1,1), (w_avg,h_avg), color, 1)

def drawCards(img, rects, maybe=None, nope=None):
  print(type(img), "\t", type(rects))
  if not(img.any()):
    return 0

  if img.dtype != 'uint8':
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


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

def findCards(img):
  # Convert to Grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Blur to reduce noise
  gray = cv2.blur(gray, (5,5))

  # Canny Edge Detection
  edge_thresh1 = 105
  edge_thresh2 = 210
  edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)

  # Morphological Operations (e.g erode, dilate)
  kernel = np.ones((7,7), np.uint8)   # kernel for op
  edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 1)
  #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 1)

  # Find "Contours" (e.g shapes, structures)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Find "upright" rectangles in point set (each rect is stored as (x_pos, y_pos, width, height))
  rects = [cv2.boundingRect(cnt) for cnt in contours]
  cards = []
  maybe = []
  nope = []

  if rects:
    rects = np.array(rects)
    cutoff = int( len(rects) * TOP)  # cutoff index for top (1-TOP) %

    # Get rectangle stats above cutoff
    areas = [rect[2]*rect[3] for rect in rects]
    areas = sorted( areas )[cutoff:]
    widths = sorted( rects[:,2] )[cutoff:]
    heights = sorted( rects[:,3] )[cutoff:]

    # Get average rectangle stats
    #a_avg = stats.mode(areas)[0]
    a_avg = np.mean(areas)
    w_avg = int( np.median(widths) )
    h_avg = int( np.median(heights) )
    # print("w_avg: ", w_avg, "\th_avg: ", h_avg)

    for i, rect in enumerate(rects):
      x = rect[0]   # top right corner x
      y = rect[1]   # top left corner y
      w = rect[2]   # rectangle width
      h = rect[3]   # rectangle height
      x2 = x+w      # bottom right corner x
      y2 = y+h      # bottom left corner x
      ratio = w/h   # rect w to h ratio

      # If Rectangle has good w and h, take a look
      if goodSize(w,h,w_avg,h_avg,0.2) :
        # If Rectangle has good ratio, consider it a card
        if goodRatio(ratio,0.1) :
          cards.append(rect)
        else:
          maybe.append(rect)
      else :
        nope.append(rect)

    return cards, maybe, nope, edges
  else :
    return [], [], [], []

def do_image():
  img = cv2.imread("magic1.png")
  cards, maybe, nope, mask = findCards(img)
  drawCards(img, cards, maybe, nope)
  #drawCards(mask, cards, maybe)

  # Resize frame and visualize
  #img = cv2.resize(img, None, fx = .5, fy = .5)
  #edges = cv2.resize(edges, None, fx = .5, fy = .5)

  # cv2.imwrite("med_thresh1_105.png", img)
  cv2.imshow("Source", img)
  #cv2.imshow("Edges", mask)
  cv2.waitKey(0)
  return 0

def do_video():
  # Read Video
  cap = cv2.VideoCapture("test.mp4")
  print(cap.isOpened())

  # Dev: Skip to frame 600
  for i in range(600):
    retrieve, frame = cap.read()

  while cap.isOpened():  
    # Get Frame
    retrieve, frame = cap.read()
    # print("\nretrieve? ", retrieve)
    # Get frame resolution,
    # print(cap.get(3), "\t", cap.get(4))

    #Frame to grayscale:


    # Detect cards
    cards, maybe, nope, mask = findCards(frame) # list of rect/cards in frame
    drawCards(frame, cards)
    drawCards(mask, cards)


    # Classify cards
    # if rect is on the list, keep class
    # if rect is new, run classifier
    #
    # return coordinates to extension

    # # Resize frame and visualize
    # frame = cv2.resize(frame, None, fx = .5, fy = .5)
    # edges = cv2.resize(edges3, None, fx = .5, fy = .5)
    cv2.imshow("Source", frame)
    cv2.imshow("Edges", mask)
    # cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
  return 0

def main():
  # do_image()
  
  do_video()
  return 0

main()

# Debugging: Visuals with Trackbar (https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html)
# cv2.namedWindow("Grayscale")
# cv2.imshow("Grayscale", gray)

# max_thresh = 255
# cv2.createTrackbar("Thresh", "Grayscale", edge_thresh1, max_thresh, thresh_callback)
# thresh_callback(edge_thresh)