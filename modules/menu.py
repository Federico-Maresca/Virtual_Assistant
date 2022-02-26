
from lib2to3.pgen2.token import STRING
from re import A, X
import cv2
import os
import time
import modules.functions as f
from threading import Thread, Lock
from collections import deque
from functools import partial

""" Tabella di verità per la gestione dei gesti corretti o errati nel menu corrente
"""
menuBool = [[True,  True,  True,   True,  True,  True,  True,  True,  True,  True], #Menu Principale
            [False, False,  False, False, False, False, False, False, False, False],   #dummy line
            [False, False,  False, False, False, False, False, False, False, False],   #dummy line
            [True,  True,  True,   False, False, False, False, False, False, False], #Rotazione
            [True,  True,  True,   False, False, False, False, False, False, False], #Saturazione
            [True,  True,  True,   False, False, False, False, False, False, False], #Luminosità
            [True,  True,  True,   False, False, False, False, False, False, False], #Contrasto
            [False, False,  True,  False, False, False, False, False, False, False], #Gesti
            [True,  True,  True,   False, False, False, False, False, False, False]] #Filtri

#dizionario usato per associare il numero di menu al suo nome
menuNumber = { 0 :'principale'  ,
               3 : 'rotazione'  ,
               4 : 'saturazione',
               5 : 'luminosita' ,
               6 : 'contrasto'  ,
               7 : 'gesti'      ,
               8 :'filtri'     ,
}
#dizionario usato per decidere se è necessario scrivere sopra l'immagine menu
# il contatore di intesità o rotazione#
menuWriter = { 'principale' : False,
            'saturazione' : True,
            'contrasto' : True,
            'rotazione' :True,
            'luminosita' : True,
            'gesti' : False,
            'filtri' : False
}
# Lista di funzioni filtri#
option = [ f.normal,
           f.sepia,
           f.Blur,
           f.HDR,
           f.invert,
           f.emboss,
           f.cartoon]

"""Classe Menu, qui si gestisce la logica dietro il menu e le modifiche alle immagini"""
@f.singleton
class MenuGesti():
    #Lista di immagini usate per la GUI
    # #
    imgSuccessive =  cv2.imread('GuiImages/immagini_successive.jpeg')
    imgPrecedenti =  cv2.imread('GuiImages/immagini_precedenti.jpeg')
    imgErrore =      cv2.imread('GuiImages/e3.png')
    imgGesto =       cv2.imread('GuiImages/gesture.jpeg')
    imgEseguiGesto = cv2.imread('GuiImages/gesto.jpeg')
    imgOk =          cv2.imread('GuiImages/ok.png')
    imgRuota =       cv2.imread('GuiImages/rotazione.jpeg')
    imgSaturazione = cv2.imread('GuiImages/saturazione.jpeg')
    imgContrasto =   cv2.imread('GuiImages/contrasto.jpeg')
    imgLuminosita =  cv2.imread('GuiImages/luminosita.jpeg')
    imgIndietro =    cv2.imread('GuiImages/indietro.jpeg')
    imgMenuGesti =   cv2.imread('GuiImages/menu_gesti_prova.jpeg')
    imgFiltri =      cv2.imread('GuiImages/filtri.jpeg')

    #Dizionario che associa il numero di menu alla sua immagine
    menuImg = { 
                    3 : imgRuota,
                    4 : imgSaturazione,
                    5 : imgLuminosita,
                    6 : imgContrasto,
                    7 : imgIndietro,
                    8 : imgFiltri
    }

    def __init__ (self, gestureQueue, loadPath = 'Images/', savePath = '.'):
        #static variables 
        self.currMenu = 0 # current menu
        self.okSleep=1 #sleep timers for visual feedback (amount of time to display check or x mark)
        self.errorSleep=1
        self.immaginiNextSleep=1
        self.salvataggioSleep=1
        self.limite=f.limite #maximum amount of intensity for image editing (example max +4 saturazione)
        self.count = 0
        self.filterCount = 0 #used inside filter logic as itensity
        self.first = True
        # font scrittura
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.org = (20, 300)
        self.fontScale = 1
        self.color = (0, 0, 0)
        self.thickness = 2

        self.started = False #rotating buffer of input images that can be edited
        self.immagini=deque()
        self.path = loadPath
        self.savePath = savePath
        for filename in os.listdir(self.path): #lista di tuple : [immagine, nome]
            self.immagini.append( [cv2.imread(self.path + filename, cv2.IMREAD_COLOR), filename] )

        self.filters = deque() #rotating buffer of filters
        for function in option :
            self.filters.append(function)

        #inizializza le immagini della finestra
        self.overlay = self.imgGesto.copy()
        self.feedback = self.imgEseguiGesto.copy()
        self.gesture = -1
        self.gestureQueue = gestureQueue
        self.finestraLock = Lock()
        self.img = self.getCurrImg()
    """ Usato da capture.py per ottenere le immagini della GUI che vengono cambiate in questa classe. Lock protected per permettere
    la concorrenza tra thread. 
    """
    def getOverlayFeedbackImage (self) :
        self.finestraLock.acquire()
        o = self.overlay.copy()
        f = self.feedback.copy()
        i = self.img.copy()
        self.finestraLock.release()
        return o, f, i
    """Ritorna una copia  dell'immagine corrente su cui si sta lavorando (diverso da self.img che è l'immagine temporanea in corso di modifica)

    """
    def getCurrImg(self) :
        return cv2.resize(self.immagini[0][0].copy(), (1080, 720)) #temp
    
    """
    Sovrascrive l'immagine temporanea sul buffer di immagini da modificare
    """
    def setImmagine(self) :
        self.finestraLock.acquire()
        self.immagini[0][0] = self.img.copy()
        self.finestraLock.release()

    """ Questa funzione gestisce i passaggi da un menù all'altro usando controlli sulla variabile del menù corrente
    """
    def innerMenuHandler(self) :
        if self.currMenu == 0 :
            if  self.gesture <= 1 :
                self.cambiaImmagine()
            elif self.gesture == 2 :
                print("\nSalvataggio Immagine")
                cv2.imwrite(os.path.join(self.savePath,self.immagini[0][1]), self.getCurrImg())
            elif self.gesture >=3 and self.gesture <= 6 : #entra in quel menu
                if self.gesture == 3 :
                    self.limite = f.value_num_rotations
                else :
                    self.limite = f.limite
                self.modificaImmagine()
            elif self.gesture == 7: #menu gesti
                func = partial( cv2.resize,  self.imgMenuGesti, (1080, 720) )
                self.currMenu = self.gesture
                self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, None, self.imgIndietro, None, func)
            elif self.gesture == 8 :  #menu filtri
                self.currMenu = self.gesture
                self.limite = f.limite
                self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, self.imgFiltri, None, True)
            elif self.gesture == 9 :#handle exit
                print("\n Uscita dal Programma")
                self.stop()
        # Le variabili funcPlus e funcMinus sono usate per salvare un puntatore a funzione che verrà usato dalla funzione modificaImmagine per applicare o il gesto Più o il gesto Meno#
        elif self.currMenu == 3 : #Rotazione
            if (self.count+1) == self.limite and self.gesture == 0 : #reset rotation when it reaches 360 degrees
                self.count = -1 #-1 or +1 so that rotation functions receives 0 degree rotation (selfcount++ or --)
            elif self.count-1 == -self.limite and self.gesture == 1 :
                self.count = 1
            funcPlus = partial(f.functionR, self.getCurrImg(), True, abs(self.count+1)  )
            funcMinus = partial(f.functionR, self.getCurrImg(), False, abs(self.count-1) )
            self.modificaImmagine(funcPlus, funcMinus)
        elif self.currMenu == 4 : #Saturazione
            funcPlus = partial(f.functionW, self.getCurrImg(), True, abs(self.count+1), f.saturazione )
            funcMinus =partial(f.functionW, self.getCurrImg(), False, abs(self.count-1), f.saturazione )
            self.modificaImmagine(funcPlus, funcMinus)
        elif self.currMenu == 5 : #Luminosita
            funcPlus = partial( f.functionW, self.getCurrImg(), True, abs(self.count+1), f.luminosita  ) 
            funcMinus = partial(f.functionW, self.getCurrImg(), False, abs(self.count-1), f.luminosita  ) 
            self.modificaImmagine(funcPlus, funcMinus)
        elif self.currMenu == 6 : #Contrasto
            funcPlus = partial(f.functionW, self.getCurrImg(), True, abs(self.count+1), f.contrasto  )
            funcMinus = partial(f.functionW, self.getCurrImg(), False, abs(self.count-1), f.contrasto  )
            self.modificaImmagine(funcPlus, funcMinus)

        elif self.currMenu == 7 : #Menu Gesti #esci dal menu gesti
            self.currMenu = 0
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, None,
                                                        self.imgGesto, None, self.getCurrImg )
        elif self.currMenu == 8 : #Menu Filtri
            self.cambiaFiltro()
    """Funzione principale della Classe Menu, qui effettua la wait finchè non riesce a fare pop dalla coda di gesti e continua finchè non
    riceve segnale di uscita (la variabile self.started è posta a False ed esce)"""
    def menuHandler(self) :
        while self.started :
            self.gesture = self.gestureQueue.dequeue()
            #Qui si usa la tabella di verità per controllare gesti permessi o  meno
            if menuBool[self.currMenu][self.gesture] : #gesto permesso
                self.innerMenuHandler()
            else: #gesto non permesso, error handling 
                self.modificaImmagineInner(self.errorSleep, self.imgErrore, self.imgEseguiGesto)
        self.stop()

    """Questa funzione è usata per cambiare i filtri, se è la prima volta che si entra si sceglie il filtro, le volte successive si andrà con i gesti Più o Meno
    a cambiare l'intensità del filtro da 0 a self.limite. 
    """
    def cambiaFiltro(self):
        if self.first : #prima volta, scelta del filtro da applicare
            if self.gesture == 2 : #entra in intensità
                self.first = False
                self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto)
                return
            elif self.gesture == 0 :
                rotation = 1
            elif self.gesture == 1:
                rotation = -1
            #mantengo la distanza dall'indice 0 del buffer
            self.count += rotation
            if abs(self.count) == 8 : 
                self.count = 0
            self.filters.rotate(rotation)
            #codice per applicare filtro a immagine corrente nella coda (non lo salva)
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, self.menuImg[self.currMenu],
                                        None, True, partial(f.functionF, self.getCurrImg(), self.filterCount, self.filters[0]) )
        else : #non prima volta, scelta dell'intensità del filtro
            if self.gesture == 2 : #esci dal filtro
                self.first = True
                self.currMenu = 0
                self.setImmagine()
                self.filters.rotate(-self.count)
                self.filterCount = 0
                self.count = 0
                self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, None, self.imgGesto)
                return
            elif self.gesture == 0 and self.filterCount < self.limite:
                self.filterCount += 1
            elif self.gesture == 1 and self.filterCount > 0:
                self.filterCount -= 1
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, self.menuImg[self.currMenu],
                                    None, True, partial(f.functionF, self.getCurrImg(), self.filterCount, self.filters[0]) )

    """Questa funzione effettua il cambio dell'immagine nel menù principale, ruota il buffer circolare a dx o sx a seconda del gesto Più/Meno che riceve"""
    def cambiaImmagine(self):
        if self.gesture == 0 :
            rotation = 1
        elif self.gesture == 1:
            rotation = -1
        #codice per passare a immagine successiva o precedente
        self.immagini.rotate(rotation)
        self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, None,
                                    None, None, self.getCurrImg) 

    """Usata da ogni altra funzione qui gestisce ogni cambio di immagine della finestra GUI nel modo seguente:
        -proteggi con un Lock le modifiche
        - cambia immagine di Overlay1 se la riceve
        -cambia immagine Feedback1 (per esempio mostra un check verde)
        -applica la funzione se la riceve
        -scrivi sopra overlay l'intensità alla quale si sta lavorando (per esempio scrive sopra l'overlay di saturazione : count +2)
        -rilascia il lock così che capture possa usare la GUI appena modificata

        -sleep per sleepTimer
        
        -proteggi con un Lock le modifiche
        -cambia immagine di Overlay2 se la riceve (per esempio se ho appena ricevuto il comando di uscire da un menù devo tornare al menù principale)
        -cambia immagine di Feedback2 (dopo il check verde devo ritornare al feedback standard)
        -rilascia il lock così che capture possa usare la GUI appena modificata
    
    """
    def modificaImmagineInner(self, sleepTimer , path1Feedback, path2Feedback, 
                        path1Overlay = None, path2Overlay = None, menuName = None, func = None) :
        self.finestraLock.acquire()
        if path1Overlay is not None :
            self.overlay = path1Overlay.copy()
        self.feedback = path1Feedback.copy()
        if func is not None :
            self.img= func()
        if menuName is not None: #if necessary update the counter overlay
            self.writer()
        self.finestraLock.release()

        time.sleep(sleepTimer)

        self.finestraLock.acquire()
        if path2Overlay is not None :
            self.overlay = path2Overlay.copy()
        self.feedback = path2Feedback.copy()
        self.finestraLock.release()
    """Funzione wrapper per modificaImmagineInner, decide quali funzioni e immagini di overlay e feedback inviarle"""
    #gestisce Saturazione Rotazione Contrasto Luminosita
    def modificaImmagine(self, funcPlus = None, funcDown = None) :
        #prendi valore per funzione Writer e inizializza contatore
        if funcPlus is None: #entra nel menu
            print(self.gesture)
            self.currMenu = self.gesture
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, 
                                self.menuImg[self.gesture], None, True)
            return
        #già dentro il menu
        if self.gesture == 0 and self.count < self.limite: #azione +
            self.count += 1
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, self.menuImg[self.currMenu], None, True,
                                funcPlus)
        elif self.gesture == 1 and self.count > -self.limite : #azione -
            self.count -= 1
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, self.menuImg[self.currMenu], None, True,
                                funcDown)
        elif self.gesture == 2 :  #azione conferma 
            self.count = 0
            self.currMenu = 0
            self.setImmagine()
            self.modificaImmagineInner(self.okSleep, self.imgOk, self.imgEseguiGesto, None, self.imgGesto)
    """Semplice funzione writer che scrive sopra overlay per indicare l'intensità delle modifiche applicate"""
    def writer(self):
        if self.currMenu == 8 : #son nel menu filtri
            var = self.filters[0].__name__
            tmp = self.filterCount
        else :
            var = menuNumber[self.currMenu]
            tmp = self.count
        if tmp > 0 :
            if tmp == self.limite :
                var = var + " max"
            else :
                var = var + " +" + str(tmp)
        elif tmp < 0:
            if tmp == -self.limite :
                var = var + " min"
            else :
                var = var +  " " + str(tmp)
        else :
                var = var +" 0"
        cv2.putText(self.overlay, var, self.org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
    
    
    def start(self) :
        if self.started :
            print ("Thread has already been started.\n")
            return None
        self.started = True
        self.menuHandler()
        return self

    def stop(self) :
        self.started = False
        time.sleep(1)
 
