import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from threading import Thread
import time
# Import utilites
from tensorflow.keras.models import load_model
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as viz_utils
from builders import model_builder
from utils import config_util

paths = {
    'CHECKPOINT_PATH': os.path.join('my_ssd_mobnet'), 
 }

files = {
    'PIPELINE_CONFIG':os.path.join('my_ssd_mobnet', 'pipeline.config'),
    #'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join('my_ssd_mobnet', 'label_map.pbtxt')
}



def hconcat_resize(img_list, interpolation =cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation
                                 =interpolation)
                      for img in img_list]

    # return final image
    return cv2.hconcat(im_list_resize)

def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)

def concat_tile_resize(list_2d, interpolation=cv2.INTER_CUBIC):
    # function calling for every
    # list of images
    img_list_v = [hconcat_resize(list_h,interpolation=cv2.INTER_CUBIC) for list_h in list_2d]

    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)



@tf.function
def detect_fn(image,detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def CaptureCamera ():
    global ans
    global kill_capture
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)


    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-8')).expect_partial()
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    

    vid = cv2.VideoCapture(1,cv2.CAP_DSHOW)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture_stop=True
    while (kill_capture==False):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        frame_resized = cv2.resize(frame, (540, 360))

            #if capture_stop == False :
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor,detection_model)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.85,
                    agnostic_mode=False)

        max_val = max(detections['detection_scores'])
        if (max_val > 0.85) :
            max_idx = np.where(detections['detection_scores']==max_val)
            ans = detections['detection_classes'].__getitem__(max_idx)
        else :
            ans = -1
        #capture_stop = True
        
        
        # function calling
        im_tile_resize = concat_tile_resize([[overlay, frame_resized, feedback],
                                                  [img]])
        # show the image
        cv2.imshow('detections',image_np_with_detections)
        cv2.imshow('Computer_vision_Project', im_tile_resize)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    # After the loop release the cap object
    vid.release()

def writer( controllo):
    if (count_up > count_down):
        if (count_up == limite):
            cv2.putText(overlay, controllo+" max", org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(overlay, controllo+" +" + str(count_up), org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif (count_up < count_down):
        if (count_down == limite):
            cv2.putText(overlay, controllo+" min", org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(overlay, controllo+" -" + str(count_down), org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(overlay, controllo+" 0", org, font, fontScale, color, thickness, cv2.LINE_AA)

def saturazione(soglia):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    if(soglia):
     s = s * value_saturazione
    else:
     s = s / value_saturazione
    s = np.clip(s, 0, 255)
    return cv2.merge([h, s, v])

def rotazione(soglia):

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

def luminosita(soglia):
    v = img[:, :, 2]
    if(soglia):
      return  np.where(v <= 255 - value_luminosita, v + value_luminosita, 255)
    else:
      return np.where(v >= value_luminosita, v - value_luminosita, 0)

def sepia():
    img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))  # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
    return np.array(img_sepia, dtype=np.uint8)

def gaussianBlur():
   '''
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16

    return cv2.filter2D(img, -1, kernel)
    '''
   return  cv2.GaussianBlur(img , (15,15) , 0)

def HDR():
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

def invert():
    return cv2.bitwise_not(img)

def emboss():
    kernel = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
    return cv2.filter2D(img, -1, kernel)

def pencil_sketch_color():
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_color

def edge_detection():

    kernel = np.array([[-1,-1,-1],
                            [-1,8,-1],
                            [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def writer_filtri( controllo):
        cv2.putText(overlay, controllo, org, font, fontScale, color, thickness, cv2.LINE_AA)

#carica in path la cartella dove si trovano le immagini , e mette nella lista immagini tutti i nomi delle immagini presenti
immagini=[]
location=0
path = 'Images/'
for filename in os.listdir(path):
    immagini.append(filename)

#inizializza le immagini della finestra
img = cv2.imread(path + immagini[location], cv2.IMREAD_COLOR)
img = cv2.resize(img, (1080, 720))
overlay = cv2.imread('immagini_progetto/gesture.jpeg')
feedback = cv2.imread('immagini_progetto/gesto.jpeg')

#valori variabili varie

ok_sleep=1
error_sleep=1
immagini_next_sleep=1
salvataggio_sleep=1
limite=4
count_up = 0
count_down = 0

value_saturazione=1.3
value1_alpha_contrasto = 1.15
value2_alpha_contrasto= 0.85
value_beta_contrasto = 0
value_luminosita=10
value_rotation=90

# font scrittura
font = cv2.FONT_HERSHEY_DUPLEX
org = (20, 300)
fontScale = 1
color = (0, 0, 0)
thickness = 2


global ans
global capture_stop 
global kill_capture
kill_capture = False
ans = -1
capture_stop = False
#parte thread camera , che gestisce anche finestra applicazione
c=Thread(target=CaptureCamera)
c.start()


ans1= "0"
gesti= {"visualizza gesti" : 7, "saturazione" : 4, "contrasto" : 6, "luminosita" :5, "ruota" : 3, "filtri" : 8,"+" :0 , "-" : 1, "conferma" : 2, "esci" : 9}
time.sleep(10)
if(capture_stop==False) :
    time.sleep(20)

while ans!=9:
    overlay = cv2.imread('immagini_progetto/gesture.jpeg')
    print(ans)

    #input deve essere sostiuito da funzione che cattura il gesto e lo trasforma in un intero , ad ogni intero corrisponde un gesto diverso
    if ans==-1 :
        print("Test -1\n")
        continue
    elif ans == 0:

      #codice per passare a immagine successiva se esiste
      if immagini.__len__()==location+1 :
          overlay = cv2.imread('immagini_progetto/immagini_successive.jpeg')
          time.sleep(immagini_next_sleep)
          overlay = cv2.imread('immagini_progetto/gesture.jpeg')
          continue

      feedback = cv2.imread('immagini_progetto/ok.png')
      location=location+1
      img = cv2.imread(path + immagini[location], cv2.IMREAD_COLOR)
      img = cv2.resize(img, (1080, 720))
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
    
    elif ans == 1:

      # codice per passare a immagine precedente se esiste
      if location==0:
          overlay = cv2.imread('immagini_progetto/immagini_precedenti.jpeg')
          time.sleep(immagini_next_sleep)
          overlay = cv2.imread('immagini_progetto/gesture.jpeg')
          continue

      feedback = cv2.imread('immagini_progetto/ok.png')
      location = location - 1
      img = cv2.imread(path + immagini[location], cv2.IMREAD_COLOR)
      img = cv2.resize(img, (1080, 720))
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

    elif ans == 2:

      overlay = cv2.imread('immagini_progetto/salvataggio.jpeg')
      time.sleep(salvataggio_sleep)
      #codice per salvare foto

    elif ans==3:

      overlay = cv2.imread('immagini_progetto/rotazione.jpeg')
      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      while ans != 2:

          if ans == 0:

              feedback = cv2.imread('immagini_progetto/ok.png')
              img = rotazione(True)
              time.sleep(ok_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')
              overlay = cv2.imread('immagini_progetto/rotazione.jpeg')

          elif ans == 1:

              feedback = cv2.imread('immagini_progetto/ok.png')
              img = rotazione(False)
              time.sleep(ok_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')
              overlay = cv2.imread('immagini_progetto/rotazione.jpeg')

          else:
              feedback = cv2.imread('immagini_progetto/e3.png')
              time.sleep(error_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      ans = -1
      continue

    elif ans == 4:

      overlay = cv2.imread('immagini_progetto/saturazione.jpeg')
      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
      cv2.putText(overlay, "saturazione" + " 0", org, font, fontScale, color, thickness, cv2.LINE_AA)

      count_up = 0
      count_down = 0

      while ans != 2:

          if ans == 0 and count_up<limite:

                feedback = cv2.imread('immagini_progetto/ok.png')
                img=cv2.cvtColor(saturazione(True).astype("uint8"), cv2.COLOR_HSV2BGR)
                count_up += 1
                count_down -= 1
                time.sleep(ok_sleep)
                feedback = cv2.imread('immagini_progetto/gesto.jpeg')
                overlay = cv2.imread('immagini_progetto/saturazione.jpeg')

          elif ans== 1 and count_down<limite:

                feedback = cv2.imread('immagini_progetto/ok.png')
                img = cv2.cvtColor(saturazione(False).astype("uint8"), cv2.COLOR_HSV2BGR)
                count_up -= 1
                count_down += 1
                time.sleep(ok_sleep)
                feedback = cv2.imread('immagini_progetto/gesto.jpeg')
                overlay = cv2.imread('immagini_progetto/saturazione.jpeg')

          else :
              feedback = cv2.imread('immagini_progetto/e3.png')
              time.sleep(error_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')


          writer("saturazione")


      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
      ans = 0
      continue

    elif ans==5:

      overlay = cv2.imread('immagini_progetto/luminosita.jpeg')
      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
      cv2.putText(overlay, "luminosita'" + " 0", org, font, fontScale, color, thickness, cv2.LINE_AA)

      count_up=0
      count_down=0

      while ans != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if ans == 0 and count_up <limite:

              feedback = cv2.imread('immagini_progetto/ok.png')
              img[:, :, 2] = luminosita(True)
              img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
              count_up +=1
              count_down -=1
              time.sleep(ok_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')
              overlay = cv2.imread('immagini_progetto/luminosita.jpeg')

        elif ans == 1 and count_down <limite:

              feedback = cv2.imread('immagini_progetto/ok.png')
              img[:, :, 2] = luminosita(False)
              img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
              count_down +=1
              count_up-=1
              time.sleep(ok_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')
              overlay = cv2.imread('immagini_progetto/luminosita.jpeg')

        else:
              img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
              feedback = cv2.imread('immagini_progetto/e3.png')
              time.sleep(error_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')

        writer("luminosita'")


      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      ans = 0
      continue

    elif ans==6:

      overlay = cv2.imread('immagini_progetto/contrasto.jpeg')
      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
      cv2.putText(overlay, "contrasto" + " 0", org, font, fontScale, color, thickness, cv2.LINE_AA)

      count_up = 0
      count_down = 0
      #start capture
      # stop capture  
      while ans != 2:

          if ans == 0 and count_up<limite:

                feedback = cv2.imread('immagini_progetto/ok.png')
                img= cv2.convertScaleAbs(img, alpha=value1_alpha_contrasto, beta=value_beta_contrasto)
                count_up += 1
                count_down -= 1
                time.sleep(ok_sleep)
                feedback = cv2.imread('immagini_progetto/gesto.jpeg')
                overlay = cv2.imread('immagini_progetto/contrasto.jpeg')

          elif ans1 == 1 and count_down <limite:

                feedback = cv2.imread('immagini_progetto/ok.png')
                img = cv2.convertScaleAbs(img, alpha=value2_alpha_contrasto, beta=value_beta_contrasto)
                count_up -= 1
                count_down += 1
                time.sleep(ok_sleep)
                feedback = cv2.imread('immagini_progetto/gesto.jpeg')
                overlay = cv2.imread('immagini_progetto/contrasto.jpeg')

          else:
              feedback = cv2.imread('immagini_progetto/e3.png')
              time.sleep(error_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')


          writer("contrasto")

      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      ans = 0
      continue
    
    elif ans==7:
      capture_stop=True
      #parte la schermata che fa visualizzare tutti i gesti possibili

      overlay = cv2.imread('immagini_progetto/indietro.jpeg')
      feedback = cv2.imread('immagini_progetto/ok.png')

      img= cv2.imread('immagini_progetto/menu_gesti_prova.jpeg')
      img = cv2.resize(img, (1080, 720))
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      time.sleep(ok_sleep)
      while ans!= 2:
          feedback = cv2.imread('immagini_progetto/e3.png')
          time.sleep(error_sleep)
          feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      feedback = cv2.imread('immagini_progetto/ok.png')


      img = cv2.imread(path + immagini[location], cv2.IMREAD_COLOR)
      img = cv2.resize(img, (1080, 720))
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
      capture_stop=False
      continue
 
    elif ans ==8:
      print("\napri modalita filtri")

      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')
      overlay=cv2.imread('immagini_progetto/filtri.jpeg')

      img_backup = img

      options = {0: sepia,
                 1: gaussianBlur,
                 2: HDR,
                 3: invert,
                 4: emboss,
                 5: pencil_sketch_color,
                 6: edge_detection ,
                 }

      filtro = 0
      img=options[filtro]()
      writer_filtri(options.get(filtro).__name__)

      print("scegli che filtro vuoi applicare")

      while ans != 2:

          if ans == 0 :

              if(filtro>=(options.__len__()-1)):
                  overlay = cv2.imread('immagini_progetto/filtri_successivi.jpeg')
                  time.sleep(immagini_next_sleep)
                  overlay = cv2.imread('immagini_progetto/filtri.jpeg')

              else:
                  print("filtro successivo")
                  filtro+=1
                  img = img_backup
                  img=options[filtro]()
                  feedback = cv2.imread('immagini_progetto/ok.png')
                  time.sleep(ok_sleep)
                  feedback = cv2.imread('immagini_progetto/gesto.jpeg')
                  overlay = cv2.imread('immagini_progetto/filtri.jpeg')

          elif ans == 1 :

              if(filtro <=0):
                  overlay = cv2.imread('immagini_progetto/filtri_precedenti.jpeg')
                  time.sleep(immagini_next_sleep)
                  overlay = cv2.imread('immagini_progetto/filtri.jpeg')
              else:
                  print("filtro precedente")
                  filtro -= 1
                  img = img_backup
                  img=options[filtro]()
                  feedback = cv2.imread('immagini_progetto/ok.png')
                  time.sleep(ok_sleep)
                  feedback = cv2.imread('immagini_progetto/gesto.jpeg')
                  overlay = cv2.imread('immagini_progetto/filtri.jpeg')

          else:

              feedback = cv2.imread('immagini_progetto/e3.png')
              time.sleep(error_sleep)
              feedback = cv2.imread('immagini_progetto/gesto.jpeg')
              overlay = cv2.imread('immagini_progetto/filtri.jpeg')

          writer_filtri(options.get(filtro).__name__)
          ans1 = input("What would you like to do?")


      feedback = cv2.imread('immagini_progetto/ok.png')
      time.sleep(ok_sleep)
      feedback = cv2.imread('immagini_progetto/gesto.jpeg')

      ans = 0
      continue

    elif ans == 9:
      print("\nesci")
      #bisogna chiudere thread immagine
      #bisogna salvare qualcosa prima di uscire?
      #c=0
      kill_capture = True
      time.sleep(5)
      # After the loop release the cap object
      c.join()
      cv2.destroyAllWindows()

    else :

        feedback = cv2.imread('immagini_progetto/e3.png')
        time.sleep(error_sleep)
        feedback = cv2.imread('immagini_progetto/gesto.jpeg')



