<div id="top"></div>
<br />
<div align="center">

  <h3 align="center">VISUAL ASSISTANCE IN IMAGE EDITING</h3>
  <h4 align="center">Federico Maresca, Mattia Manfredini, Francesco Giacalone</h4>
  <p align="center">
    A Visual assistant for gesture recognition in the editing of images
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

The neural network was trained with Tensorflow and the image editing was written using OpenCV.

* [OpenCV](https://opencv.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Python](https://https://www.python.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This python program requires Tensorflow and some setup so we suggest using a virtualenv with pip. All requirements are saved in the [requirements.txt](https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/requirements.txt) file


### Installation

_These instructions allow for the quick setup and installation of all dependencies such as the Tensorflow model zoo and various libraries._

1. Clone the repo.
   ```sh
   git clone https://github.com/Federico-Maresca/Virtual_Assistant.git
   ```
2.  Clone the Tensorflow repo inside our program folder.
   ```sh
   git clone https://github.com/tensorflow/models.git
   ```
### Your folder structure should now look somethink like this:

    Virtual_Assistant
    ├── models                   # This is the tensorflow repo
    ├── SSD_Network                    # Trained SSD model
    ├── Immagini                     #  Other folders
    ├── etc...                    
    
3. Install virtualenv (skip if you already have virtualenv package installed)
   ```sh
   pip install virtualenv 
   ```
4. Generate a virtual environment and access it
   ```sh
   python -m virtualenv env && source env/bin/activate
   ```
5. Install protobuf (```sudo apt install protobug-compiler``` in Ubuntu) and run it to generate all the necessary .py files in the models/research/object_detection/protos folder (the end of this command returns you to the starting folder
   ```sh
   cd models/research && protoc object_detection/protos/*.proto --python_out=. && cd ../..
   ```
6. Modify the [requirements.txt](https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/requirements.txt) file so that the object_detection module path points to the research folder inside the models folder as shown inside the file (line 49) and move the setup.py file 
     ```sh
   cp models/research/object_detection/packages/tf2/setup.py ./models/research
   ```
8.  Run the following command to install all necessary libraries and dependencies
   ```sh
   pip install -r requirements.txt
   ```
8. Now all is ready to use, just (always from your virtual  environment) run the following command line command
   ```sh
   python3 main.py
   ```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

