## Run these commands to install dlib:
brew install cmake python
brew install dlib
brew install exiftool

python3 -m venv .venv
source .venv/bin/activate
pip3 install numpy
pip3 install dlib



## For models/ directory, download dlib HOG's *.dat files from ONCE:
mkdir -p models
if [ ! -e mmod_human_face_detector.dat ]; then
  wget -P models  https://github.com/keyurr2/face-detection/blob/master/mmod_human_face_detector.dat
  wget -P models  https://github.com/mrolarik/face-recognition-system/tree/master/dlib-face-model
fi

