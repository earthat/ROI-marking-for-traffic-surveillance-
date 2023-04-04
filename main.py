import os
import logging
import logging.handlers
import random
import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt
import collections
import utils
from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

# ============================================================================
refpoints=[]
IMAGE_DIR = "./out"
#VIDEO_SOURCE = "input.MP4"
VIDEO_SOURCE = "Khare_testvideo_02.avi"
#SHAPE = (360,640)  # HxW
'''EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])'''
'''EXIT_PTS = np.array([
    [[366, 360], [366, 295], [640, 250], [640, 360]],
    [[0, 200], [322, 200], [322, 0], [0, 0]]
])'''

#SHAPE=(1080,1920)
#EXIT_PTS=np.array([
 #   [[790,578], [885,586], [661, 603], [885, 618]],
  #  [[991, 582], [1078, 562], [1027, 542], [1080, 551]]
#])
#EXIT_PTS=np.array([[[361,248],[351,210],[469,186],[519,212]]])


# ============================================================================
def selectroi (event, x, y, flags, param):   # function to mark the entry and exit at roundabout
    #refPt=list()
    global refPt
    global videocntrl
    cnt=0

    cv2.imshow('SelectROI', firstFrame)
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt=[(x, y)]
        '''cnt = cnt + 1
        if cnt == 4:
            refpoints.append(refPt)
        else:
            refPt.append((x, y))'''


    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        #refPt.append((x, y))
        cv2.circle(firstFrame, ((x, y)), 3, (0,255, 255), -1)
        #refPt[0][0], refPt[0][1]
        # draw a rectangle around the region of interest
        #cv2.rectangle(firstFrame, refPt[0], refPt[1], (0, 255, 0), 2)
        #cv2.line(firstFrame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("SelectROI", firstFrame)
        refpoints.append(refPt)

    elif event == cv2.EVENT_RBUTTONDOWN:
        videocntrl = True
        cv2.destroyAllWindows()

def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    while True:
        _,frame=cap.read()
    #for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap




#*************main function starts here************************
#*************main function starts here************************
log = utils.init_logging()

if not os.path.exists(IMAGE_DIR):
    log.debug("Creating image directory `%s`...", IMAGE_DIR)
    os.makedirs(IMAGE_DIR)

log = logging.getLogger("main")
# Set up image source
# You can use also CV2, for some reason it not working for me
# cap = skvideo.io.vreader(VIDEO_SOURCE)
cap = cv2.VideoCapture(VIDEO_SOURCE)
_, firstFrame = cap.read()  # read the first frame for ROIs
cv2.namedWindow('SelectROI',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('SelectROI', selectroi)
cv2.waitKey(0)
pt=[]
strt=0
end=4
for i in range(int((np.shape(refpoints)[0])/4)):
    pt.append(refpoints[strt:end])
    strt=end
    end=end+4

EXIT_PTS = np.array(pt)
SHAPE =(np.shape(firstFrame)[0],np.shape(firstFrame)[1])
# creating exit mask from points, where we will be counting our vehicles
base = np.zeros(SHAPE + (3,), dtype='uint8')
exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

# there is also bgslibrary, that seems to give better BG substruction, but
# not tested it yet
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, detectShadows=True)

# processing pipline for programming conviniance
pipeline = PipelineRunner(pipeline=[
    ContourDetection(bg_subtractor=bg_subtractor,
                     save_image=True, image_dir=IMAGE_DIR),
    # we use y_weight == 2.0 because traffic are moving vertically on video
    # use x_weight == 2.0 for horizontal.
    VehicleCounter(exit_masks=[exit_mask], yS_weight=2.0),
    Visualizer(image_dir=IMAGE_DIR),
    CsvWriter(path='./', name='report.csv')
], log_level=logging.DEBUG)

# skipping 500 frames to train bg subtractor
train_bg_subtractor(bg_subtractor, cap, num=500)

_frame_number = -1
frame_number = -1
cv2.namedWindow('op', cv2.WINDOW_NORMAL)
while True:
    
    # Read the video frame by frame
    ret, frame = cap.read()
    if not ret:
        print('Error: Could not read frame from video file')
        break

    # real frame number
    _frame_number += 1

    # skip every 2nd frame to speed up processing
    if _frame_number % 2 != 0:
        continue

    # frame number that will be passed to pipline
    # this needed to make video from cutted frames
    frame_number += 1

    # plt.imshow(frame)
    # plt.show()
    # return
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pipeline.set_context({
        'frame': frame,
        'frame_number': frame_number,
    })
    cc = pipeline.run()
    frame = cc['frame']
    cv2.imshow('op', frame)
    if cv2.waitKey(33) == 27:
        break
#def main():

# ============================================================================

'''if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()'''
