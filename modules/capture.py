import time
import cv2
from threading import Thread

from modules.object_detector import ObjectDetector
from modules.object_detector import ObjectDetectorOptions
""" Classe Capture gestisce la webcam e la rete neurale. Invia i gesti tramite la classe gestureQueue che riceve durante la inizializzazione dal main

"""
class Capture :
    def __init__(self, menu, model, gestureQueue, src : int, width : int, height : int, num_threads : int, score_threshold : float ) :
        self.stream = cv2.VideoCapture(src) #capture video
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width) #set height and width
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read() #read first frame
        self.started = False #used to stop update while loop 
        self.menu = menu #used to access menu images to build the GUI
        options = ObjectDetectorOptions( #object detector options 
          num_threads=num_threads, #number of threads to use for inference
          score_threshold=score_threshold, # min class probability
          max_results=1) #number of maximum results to display
        self.model = ObjectDetector(model_path=model, gestureQueue=gestureQueue, options=options) #object detector model

    def start(self) :
        if self.started :
            print ("Thread has already been started.\n")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    """ Funzione principale di Capture. Qui il loop di cattura frame e di object detection
    """
    def update(self) :
        while self.started:
            self.grabbed, self.frame = self.stream.read() #get current frame
            
            #run detection
            self.frame = self.model.detectionW(self.frame)
            #create GUI
            self.buildWindow(self.frame)
            #sleep
            time.sleep(0.03) #~30 fps 1/30
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
    """Queste funzioni costruiscono la finestra aggiustando la dimensione delle immagini e concatenandole usando 
    le funzioni cv2.hconcat e cv2.vconcat
    """
    def concatTileResize(self, list_2d, interpolation=cv2.INTER_CUBIC):
        # function calling for every
        # list of images

        imgListV = [self.hconcatResize(listH,interpolation=cv2.INTER_CUBIC) for listH in list_2d]
                # return final image
        return self.vconcatResize(imgListV, interpolation=cv2.INTER_CUBIC)

    def hconcatResize( self, imgList, interpolation =cv2.INTER_CUBIC):
    # take minimum heights
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
    """Queste due funzioni gestiscono l'uscita del thread"""
    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()
