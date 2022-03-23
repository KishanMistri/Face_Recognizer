### Face Recognizer (using CNN)

# Short pointers on details:
- Using openCV libraby and Python language on Anaconda Platform 
- Face detection using Haar cascade Training detector 
- Dataset used: [AT&T] and [Yalefaces]
- Dimension reduction using LDA approach and recognization using CNN training on train dataset.
- Adds new face image in database while testing on new file
- Achieved 82.3% accuracy.

# How to Use:
- save/clone project at like C:\Face_Recognizer
- Now you have C:\Face_Recognizer\face_rec_demo which contains 5 .py files.
- from command line/Terminal
  - move to directory - 
  > C:\Face_Recognizer 
  - run
  > python face_rec_demo imgname dirname numofeigenfaces threshold

  | Param | Description |
  | ------ | ------ |
  | dirname | directory where images of same extension reside| 
  | numofeigenfaces | how many eigenfaces need to be used in matching(shd be less than the number of images (of same extension)in folder represented by dirname.| 
  | threshold | a threshold value that should be the upper limit of euclidean distance between images | 

    > example: to compare F:\myimages\probes\Raj3.png against png images in the folder 'F:\myimages\gallery' using 6 eigen vectors (faces) and with distance below 3

    ```sh
    python face_rec_demo F:\myimages\probes\Raj3.png F:\myimages\gallery 6 3
    ```

# Specs & testing:
- Tested on Windows7,8,10 + python 2.7
- Ubuntu Lucid + python 2.6.5


> Note: This is not the optimal solution for face recognition as there are state of the art deep learning techniques in place. This project was a study to understand the how standard/classical ML can help solve face recognition issue and steps to implement one.
> Similar approach via rasberrypie can be found [here](https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348)
