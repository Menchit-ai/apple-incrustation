# Developer's guide

## The main

The main is split in 4 parts. The first part handle all the files, folders and background's objects (such as depths) creation and parse the command write by the user. Then, it select random parameters (random background, objects, iterations...) and pass them to the incrust function. The fourth part is used after the incrustation and will be discussed at the end of the incrust chapter.

## Incrust

The incrust function will deal with all the incrustations. It does incrustation one by one and return the coordinates of the incrustation to the main function which will write those coordinates in a txt file that has the same nam as the generated image. The incrustation is done following 4 steps :

1. First, it loads all the objects related to the current background. Then, it will choose a random point in a random depth of the bacground.
2. It will scale our object depending on the chosen depth.
3. Next, it will create the enlightened object by computing where the light is arriving in the object's area, the light mask and the application of this light mask to our object.
4. Finally, it will select an area in the chosen depth (area defined by our object shape and the starting point selected at step 2). Incrust our object in that area and changing the area in the whole depth.

The coordinates are returned and the depths are written in a file correponding to the generated image file's name.

Now we have depths that are images containing the incrusted objects and a txt file containing all the coordinates of incrusted object. The main just have to reconstruct the depth by adding the non black pixels of each depths, starting by the furthest one : the closet pixels will recover the furthest and create an illusion of deepness.

## Lib and wrapper

The lib folder contains basics functions that are used in the script and files that are needed to create depth. the folder netowkrs and the files layers.py and utils.py come from the repo <https://github.com/nianticlabs/monodepth2> which was used to implement the monodepth2 model which create the depth in the backgrounds. I made the wrapper called depth_model.py which is a single class that allow me to use the monodepth2 model easily with making all basics initialisation and computation inside the class object. The file library.py contains all the function that I have work on since the beginning of my project. Most of them are unused but I let them so you can see my reflexions and my progression.