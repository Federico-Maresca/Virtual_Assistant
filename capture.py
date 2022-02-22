import time
import cv2
from threading import Thread, Lock
import detection as d
class Capture :

    def __init__(self, menu, gestureQueue, src = 0, width = 1280, height = 720) :
        self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        self.menu = menu
        self.model = d.Detection(gestureQueue)

    def start(self) :
        if self.started :
            print ("Thread has already been started.\n")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started:
            self.grabbed, self.frame = self.stream.read() #get current frame
            
            #run detection
            self.frame = self.model.detectionW(self.frame)
            self.buildWindow(self.frame)
            time.sleep(0.033) #~30 fps 1/30
        cv2.destroyAllWindows()

    def buildWindow (self, frame):
        frame_resized = cv2.resize(frame, (540, 360))    
        overlay, feedback, img = self.menu.getOverlayFeedbackImage()
        # function calling
        window = self.concatTileResize([[overlay, frame_resized, feedback],
                                                  [img]])
        # show the image
        cv2.imshow('Computer_vision_Project', window)
        cv2.waitKey(1)
            
    def concatTileResize(self, list_2d, interpolation=cv2.INTER_CUBIC):
        # function calling for every
        # list of images

        imgListV = [self.hconcatResize(listH,interpolation=cv2.INTER_CUBIC) for listH in list_2d]
                # return final image
        return self.vconcatResize(imgListV, interpolation=cv2.INTER_CUBIC)

    def hconcatResize( self, imgList, interpolation =cv2.INTER_CUBIC):
    # take minimum hights
        h_min = min(img1.shape[0]
                    for img1 in imgList)
        # image resizing
        imListResize = [cv2.resize(img1,
                                     (int(img1.shape[1] * h_min / img1.shape[0]),
                                      h_min), interpolation
                                     =interpolation)
                          for img1 in imgList]
        # return final image
        return cv2.hconcat(imListResize)

    def vconcatResize( self, imgList, interpolation=cv2.INTER_CUBIC):
        # take minimum width
        w_min = min(img1.shape[1]
                    for img1 in imgList)
        # resizing images
        imListResize = [cv2.resize(img1,
                                     (w_min, int(img1.shape[0] * w_min / img1.shape[1])),
                                     interpolation=interpolation)
                          for img1 in imgList]
        # return final image
        return cv2.vconcat(imListResize)
  
    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()
