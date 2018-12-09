import piexif
import glob
import os

os.chdir("data/wikiart/")
for img in glob.glob("*.jpg"):
    piexif.remove(img)
