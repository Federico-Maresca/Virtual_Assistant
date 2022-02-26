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
    <li><a href="#examples">Roadmap</a></li>
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

### Prerequisites

This program can be run both from Windows and Linux. The setup is done with pip and so should be platform indipendent.
### Installation

_These instructions allow for the quick setup and installation of all dependencies such as the Tensorflow lite support package and various libraries._

1. Clone the repo.
   ```sh
   git clone https://github.com/Federico-Maresca/Virtual_Assistant.git
   ```
2. Run setup.py to install requirements. Change to python3 for Linux.
   ```sh
   python setup.py
   ```
3. Now all it's ready to use, just (always from your virtual  environment) run the following command line command to view the usage instructions.
   ```sh
   python3 main.py --help
   ```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage and Video Examples

Main features for image editing are:
  * Image Rotation
  * Image Saturation
  * Image Contrast
  * Image Luminosity
  * Filters

The following image shows all recognized gestures.
![alt text](https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/GuiImages/menu_gesti_prova.jpeg?raw=true)

The first 6 gestures are used to access their respective menus from the Main menu. Gestures Più and Meno gestures are used to either change the image that is being edited or change the value of the current editing menu (for example +1 saturation). Inside all menus Conferma gesture is used to exit and conferm changes or, if in the main menu , to save the image to the Saved Image folder specified at runtime. Finally Esci is used in the main menu to exit the program. 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- VIDEO EXAMPLES -->
## Examples
### How to change image in selected image folder
<a href="url"><img src="https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/Sample%20Videos/tutorial_change_image.gif" width="640" height="480" > </a>
 ### How to view gestures 
<a href="url"><img src="https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/Sample%20Videos/tutorial_gesture_view.gif" width="640" height="480" ></a>

### How to use the filters menu: select filter, then choose intensity
<a href="url"><img src="https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/Sample%20Videos/tutorial_filter_cartoon.gif" width="640" height="480"></a>

### How to rotate image
<a href="url"><img src="https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/Sample%20Videos/tutorial_rotation.gif" width="640" height="480"></a>

### How to exit
<a href="url"><img src="https://github.com/Federico-Maresca/Virtual_Assistant/blob/master/Sample%20Videos/tutorial_exit.gif" width="640" height="480"></a>


