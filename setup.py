import sys
import os
#Install github Python dependencies

if sys.platform.startswith("linux") :
  python = 'python3'
elif sys.platform.startswith("win32") :
  python = 'python'
os.system(python + ' -m pip install pip --upgrade')
os.system(python + ' -m pip install -r requirements.txt')

print("Dependencies where installed, all ready to go!")