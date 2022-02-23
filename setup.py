import os

cwd = os.getcwd()

os.system('pip install tensorflow==2.*')

os.system('git clone https://github.com/tensorflow/models.git')

os.system('pip install protobuf')

os.chdir( os.path.join(cwd,'models','research'))

os.system('protoc object_detection/protos/*.proto --python_out=.')
os.system('cp object_detection/packages/tf2/setup.py .')
os.system('python -m pip install .')
os.chdir( cwd)

os.system('python object_detection/builders/model_builder_tf2_test.py')