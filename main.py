import cv2
# Import utilites
import menu as m
import capture as c
import functions as f

def main() :
    gestureQueue = f.GestureQueue() 
    mymenu = m.MenuGesti(gestureQueue)
    capture = c.Capture(mymenu, gestureQueue).start()
    mymenu.start()
    capture.stop()

if __name__ == "__main__" :
    main()