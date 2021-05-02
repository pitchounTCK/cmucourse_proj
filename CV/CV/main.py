import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import glob

from get_vertices import get_vertices

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Process images or video')
    ap.add_argument("-i", "--images", 
                    action="store", 
                    help="path to input dataset of images in [IMAGES].jpg")
    ap.add_argument("-v", "--video", 
                    action="store_true", 
                    help="to start video capture")
    args = ap.parse_args()

    # loop over the images
    if args.images:
        #print(args.images)
        #for imagePath in glob.glob(args["images"] + "*.jpg"):
        imagePath = args.images + ".jpg"
        #print(imagePath)
        # load the image
        img = cv2.imread(imagePath)
    elif args.video:
        capture = cv2.VideoCapture(0)

        print("ESC key to capture image")

        while(True):
            ret, frame = capture.read()
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('video gray', grayFrame)
            #cv2.imshow('video original', frame)


            if cv2.waitKey(1) == 27:
                print("Capturing Image")
                imagePath = 'test_gray.jpg'
                cv2.imwrite(imagePath, grayFrame)
                break

        capture.release()
        cv2.destroyAllWindows()

        # load the captured image
        img = cv2.imread(imagePath)
    else:
        exit()

    pts = []
    pts = get_vertices(imagePath)
    print(pts)

if __name__ == "__main__":
    main()
