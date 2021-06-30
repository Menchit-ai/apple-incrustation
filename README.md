# apple-incrustation

## Purpose of this project

Object detection algorithms, such as yolo, are models that needs a lot of samples in order to train properly. And generate those datas is pretty long, data labellization requires a lot of people and is still done by human hand.

This project's purpose is to reduce the part attribute to this human hand and to automatically create a dataset using objects and backgrounds.

It will incrust the objects in the backgrounds and try to mimic a realistics photography. While incrusting the objects, the algorithm will store the area where all the objects are incrusted and write those values in a file along with the generated image. Finnally, we obtain a dataset of labellized images and the file that store the coordinates of our incrusted objects.

## Setup

The project is used through the script incrust.py runned in a terminal. To install the project, simply clone it via `git clone <https://github.com/Menchit-ai/apple-incrustation>`