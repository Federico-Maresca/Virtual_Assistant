
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils
from models.research.object_detection.builders import model_builder
from models.research.object_detection.utils import config_util

import tensorflow as tf
import cv2
import numpy as np
import os
import funzioni as f

@f.singleton
class Detection: #detection class for neural network
      
    def __init__(self, gestureQueue) :   
        self.paths = {
        'CHECKPOINT_PATH': os.path.join('my_ssd_mobnet_2'), 
        }
        self.files = {
            'PIPELINE_CONFIG':os.path.join('my_ssd_mobnet_2', 'pipeline.config'),
            #'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
            'LABELMAP': os.path.join('my_ssd_mobnet_2', 'label_map.pbtxt')
        }
            # Load pipeline config and build a detection model
        self.configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)
        # Restore checkpoint
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], 'ckpt-13')).expect_partial()
        self.category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])
      
        self.started = False
        self.detectCounter = 0
        self.prevGesture = -1
        self.currGestureCount = 0
        self.detectCounter = 0
        self.gestureQ = gestureQueue

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections
  
    def isReady(self) :
        self.detectCounter += 1
        if self.detectCounter > 5 :
            self.detectCounter = 0
            return True
        return False
    
    def detectionW(self, frame):
        currGesture, imgDetection = self.detection( frame) #run detection
        if currGesture == -1 :
            return frame
        elif currGesture == self.prevGesture :    #check for stable gesture (removes one off frame detections, reduces noisy detections)
            self.currGestureCount += 1
            if self.currGestureCount >= 30 :
                self.currGestureCount = 0
                self.gestureQ.enqueue(currGesture)
            return imgDetection
        else :          #unstable gesture, reset
            self.currGestureCount = 0
            self.prevGesture = currGesture
            return frame
        #if either the gesture changed or it still isn't stable enough
 
    def detection(self,frame):
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        image_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+1,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=1,
                    min_score_thresh=.95,
                    agnostic_mode=False)
        max_val = max(detections['detection_scores'])
        if (max_val >= 0.95) :
            max_idx = np.where(detections['detection_scores']==max_val)
            gesture = detections['detection_classes'].__getitem__(max_idx)[0]
            return gesture,  image_with_detections

        else :
            return -1, None