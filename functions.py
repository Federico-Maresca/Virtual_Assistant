import cv2
import numpy as np
import math
from threading import Condition


value_saturazione=1.3
value1_alpha_contrasto = 1.15
value2_alpha_contrasto= 0.85
value_beta_contrasto = 0
value_luminosita=10
value_rotation=90
value_num_rotations = int(360 / value_rotation )  
limite = 4
#img = None
def singleton(cls, *args, **kw):
    instances = {}
    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton
    
@singleton
class GestureQueue:

    def __init__(self):
        self.maxGestures = 6
        self.currSize = 0
        self.gestureCond = Condition()
        self.gestureQ = []

    def dequeue(self):
        self.gestureCond.acquire()
        print("Lock Acquired D")
        while self.currSize == 0:
            print("waiting D")
            self.gestureCond.wait() 
        print("Done Waiting D")
        gesture = self.gestureQ.pop(0)
        self.currSize -= 1

        self.gestureCond.notifyAll()
        self.gestureCond.release()

        return gesture
        
    def isFull(self) :
        self.gestureCond.acquire()
        t = self.currSize == self.maxGestures
        self.gestureCond.release 
        return t

    def enqueue(self, gesture):
        self.gestureCond.acquire()
        print("Lock Acquired E")
        while self.currSize == self.maxGestures:
            print("waiting E")
            self.gestureCond.wait()

        self.gestureQ.append(gesture)
        print("Done enqueing E")
        self.currSize += 1

        self.gestureCond.notifyAll()
        self.gestureCond.release()

def saturazione( img, soglia):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    if(soglia):
     s = s * value_saturazione
    else:
     s = s / value_saturazione
    s = np.clip(s, 0, 255)
    return cv2.merge([h, s, v]).astype("uint8")

def rotazione( img, soglia):

    h, w = img.shape[:2]
    img_c = (w / 2, h / 2)

    if(soglia):
      rot = cv2.getRotationMatrix2D(img_c, value_rotation, 1)
      rad = math.radians(value_rotation)
    else:
      rot = cv2.getRotationMatrix2D(img_c, -value_rotation, 1)
      rad = math.radians(-value_rotation)

    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    return cv2.warpAffine(img, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)

def luminosita( img, soglia):
    v = img[:, :, 2]
    if(soglia):
      img[:, :, 2] = np.where(v <= 255 - value_luminosita, v + value_luminosita, 255)
    else:
      img[:, :, 2] = np.where(v >= value_luminosita, v - value_luminosita, 0)
    return img

def normal(img) :
    return
    
def sepia( img):
    img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))  # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
    return np.array(img_sepia, dtype=np.uint8)

def gaussianBlur( img):
   '''
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16

    return cv2.filter2D(img, -1, kernel)
    '''
   return  cv2.GaussianBlur(img , (15,15) , 0)

def HDR( img):
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

def invert( img):
    return cv2.bitwise_not(img)

def emboss( img):
    kernel = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
    return cv2.filter2D(img, -1, kernel)

def pencil_sketch_color(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_color

def edge_detection( img):

    kernel = np.array([[-1,-1,-1],
                            [-1,8,-1],
                            [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def contrasto(img, soglia) :
    if soglia:
        alpha = value1_alpha_contrasto
    else :
        alpha = value2_alpha_contrasto
    return cv2.convertScaleAbs(img, alpha, value_beta_contrasto)

#wrapper funzioni immagine
def functionW(img, soglia, count , function) :
    for i in range(count) :
        img = function(img, soglia)
    return img

#wrapper rotazione
def functionR(img, soglia, count) :
    #ottimizza la rotazione (esempio se ruota a dx  di x e x > 180° ruota invece a sx di 360°-x°count = count%value_num_rotations
    if  count > value_num_rotations/2 :
        soglia = soglia  == False
        count = value_num_rotations-count
    for i in range(count) :
        img = rotazione(img, soglia)
    return img