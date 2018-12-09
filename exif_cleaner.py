import piexif
import glob
import os

if __name__ == "__main__":
    os.chdir("/home/ubuntu/Art-Generator/data/wikiart/")
    for img in glob.glob("*.jpg"):
        piexif.remove(img)
    print("done")
