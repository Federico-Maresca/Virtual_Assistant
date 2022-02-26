import cv2
import numpy as np
import math
from threading import Condition

"""
Valori usati dalle funzioni 
"""
value_saturazione=1.35
value1_alpha_contrasto = 1.05
value2_alpha_contrasto= 0.85
value_beta_contrasto = 0
value_luminosita=10
value_rotation=90
value_num_rotations = int(360 / value_rotation )  
#numero massimo di volte che si può applicare una modifica
limite = 4
#img = None
def singleton(cls, *args, **kw):
    instances = {}
    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton
""" Classe GestureQueue è il mezzo di comunicazione tra il thread menu e il thread capture. Capture accoda i gesti
mentre Menu effetta la deque per ottenerli. E' protetta da un lock che assicura la thread safety e mette in wait il menu mentre non
ci sono gesti.
"""
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

""" Funzione di saturazione immagine.
Args: 
    img : immagine da modificare
    soglia : se True aumenta la saturazione, 
             se False la riduce
Returns
    Immagine modificata
"""
def saturazione( img, soglia):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    if(soglia):
     s = s * value_saturazione
    else:
     s = s / value_saturazione
    s = np.clip(s, 0, 255)
    return  cv2.cvtColor((cv2.merge([h, s, v])).astype("uint8") , cv2.COLOR_HSV2BGR)

""" Funzione di contrasto immagine.
Args: 
    img : immagine da modificare
    soglia : se True aumenta il contrasto, 
             se False lo riduce
Returns
    Immagine modificata
"""
def contrasto(img, soglia) :
    if soglia:
        alpha = value1_alpha_contrasto
    else :
        alpha = value2_alpha_contrasto
    return cv2.convertScaleAbs(img, alpha = alpha, beta = value_beta_contrasto)

""" Funzione di rotazione immagine.
Args: 
    img : immagine da modificare
    soglia : se True ruota di +90°, 
             se False ruota di -90°
Returns
    Immagine modificata
"""
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

""" Funzione di luminosità immagine.
Args: 
    img : immagine da modificare
    soglia : se True aumenta la luminosità, 
             se False la riduce
Returns
    Immagine modificata
"""
def luminosita( img, soglia):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = img[:, :, 2]
    if(soglia):
      img[:, :, 2] = np.where(v <= 255 - value_luminosita, v + value_luminosita, 255)
    else:
      img[:, :, 2] = np.where(v >= value_luminosita, v - value_luminosita, 0)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

""" Funzione filtro normal. Serve come dummy function quando si 
è appena entrati nel menù filtri e non si è ancora effettuata una scelta
Args: 
    img
Returns
    Immagine non modificata
"""
def normal(img) :
    return img

""" Funzione filtro sepia
Args : 
    img :immagine da modificare
Returns :
    Immagine modificata
"""
def sepia( img):
    img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))  # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
    return np.array(img_sepia, dtype=np.uint8)

""" Funzione filtro blur
Args : 
    img :immagine da modificare
Returns :
    Immagine modificata
"""
def Blur(img ):

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.3

    return cv2.filter2D(img, -1, kernel)

""" Funzione filtro HDR
Args : 
    img :immagine da modificare
Returns :
    Immagine modificata
"""
def HDR( img):
    return cv2.detailEnhance(img, sigma_s=3, sigma_r=0.03)

""" Funzione filtro invert
Args : 
    img :immagine da modificare
    count : tipo di inversione da applicare
Returns :
    Immagine modificata
"""
def invert( img, count):

    (blue, green, red) = cv2.split(img)

    if (count == 0) :
       return cv2.merge([red, green, blue])
    if (count == 1) :
       return cv2.merge([red, blue, green])
    if (count == 2) :
       return cv2.merge([blue, red, green])
    if (count == 3) :
       return cv2.merge([green, red, blue])
    if (count == 4) :
       return cv2.merge([green, blue, red])

""" Funzione filtro emboss
Args : 
    img :immagine da modificare
Returns :
    Immagine modificata
"""
def emboss( img):
    kernel = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]]) / 1.65
    return cv2.filter2D(img, -1, kernel)

""" Funzione filtro cartoon
Args : 
    img :immagine da modificare
Returns :
    Immagine modificata
"""
def cartoon(img):
    #inbuilt function to create sketch effect in colour and greyscale
    #sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=100, sigma_r=0.1, shade_factor=0.075)
    #return  sk_color
    return cv2.stylization(img,sigma_s=200, sigma_r=0.95)


#wrapper funzioni immagine
""" Funzione weapper per applicare più volte un filtro partendo sempre dall'immagine originale
, così facendo si combatte il degrado immagine che si avrebbe lavorando su una immagine precedenemente
modificate. Per esempio per passare da una saturazione +3 a una saturazione +2 anzichè applicare la funzione di saturazione
all'immagine già modificata precedentemente si applica la saturazione due volte all'immagine originale.
Args : 
    img :immagine da modificare
    soglia : valore True o False da passsare alla funzione 'function'
    count : numero di volte da applicare il filtro
    function : puntatore a funzione da applicare
Returns :
    Immagine modificata
"""
def functionW(img, soglia, count, function) :
    for i in range(count) :
        img = function(img, soglia)
    return img

#wrapper rotazione
""" Funzione wrapper di rotazione
Args : 
    img :immagine da modificare
    soglia : valore True o False da passsare alla funzione 'rotazione'
    count : numero di rotazioni da effettuare
Returns :
    Immagine modificata
"""
def functionR(img, soglia, count) :
    #ottimizza la rotazione (esempio se ruota a dx  di x e x > 180° ruota invece a sx di 360°-x°count = count%value_num_rotations
    if  count > value_num_rotations/2 :
        soglia = soglia  == False
        count = value_num_rotations-count
    for i in range(count) :
        img = rotazione(img, soglia)
    return img

""" Funzione wrapper filtri. Inver ha un comportamento diverso dagli altri filtri.
Args : 
    img :immagine da modificare
    count : numero di volte che bisogna applicare il filtro o combinazione di invert da usare
    function : puntatore a funzione filtro da applicare
Returns :
    Immagine modificata
"""
def functionF(img, count, function) :
    if function == invert :
        return invert(img, count)
    for i in range(count) :
        img = function(img)
    return img