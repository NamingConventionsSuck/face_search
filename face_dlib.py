import dlib

import os
import sys
import logging
import time

PROG = sys.argv[0]
# A small number of known faces should be are saved
AFACE_PATH = "~/Pictures/known_faces"
AFACE_PREFIX = "aface_"


# Models for dlib face recognition. See installation instruction above
DLIB_FACE_DETECTOR_FILE = "mmod_human_face_detector.dat"     # <1MB
DLIB_SHAPE_MODEL_FILE = "shape_predictor_5_face_landmarks.dat"      # 9MB
DLIB_FACE_REC_MODEL_FILE = "dlib_face_recognition_resnet_model_v1.dat"      # 21MB

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "models")

# Load Dlib models for face detector, landmark predictor, and face recognition
# To chose what to use, watch https://youtu.be/j27xINvkMvM?t=2839
FACE_DETECTOR = dlib.get_frontal_face_detector()    # HOG detector
FACE_PREDICTOR = dlib.shape_predictor(os.path.join(MODEL_PATH, DLIB_SHAPE_MODEL_FILE))
FACE_ENCODER = dlib.face_recognition_model_v1(os.path.join(MODEL_PATH, DLIB_FACE_REC_MODEL_FILE))


logging.basicConfig(stream=sys.stderr, level=logging.INFO, datefmt="%H%M%S",
                    format='I%(asctime)s %(filename)s:%(lineno)d: %(message)s')
LOG = logging.info


def extract_faces(image_path: str, out_path: str = ""):
    """ Detect faces in image_path and align the face
        If out_path is provided, then write the aligned+cropped grayscale face images there
        Otherwise, return a list of aligned+cropped grayscale face images for further processing
    """
    LOG(f"extract_faces from {image_path}. Write faces to {out_path}" if out_path else "")
    out_path = os.path.expanduser(out_path)
    img_bname = os.path.splitext(os.path.basename(image_path))[0]
    img = dlib.load_rgb_image(image_path)

    # Detect faces in the image
    t0 = time.time()
    det_faces = FACE_DETECTOR(img)  # upsample_num_times=1 runs much slower than 0
    t_detect = time.time() - t0
    LOG(f"\tdetect {len(det_faces)} faces from {img_bname}  TOOK {t_detect}s")

    found_faces = []
    for i, face in enumerate(det_faces):
        shape = FACE_PREDICTOR(img, face)
        face_chip = dlib.get_face_chip(img, shape)
        found_faces.append(face_chip)
        if out_path:
            out_file = f"{os.path.join(out_path, img_bname)}_{AFACE_PREFIX}{i:02d}.jpg"
            LOG(f"  Save face chip to {out_file}")
            dlib.save_image(face_chip, out_file)
            continue

    t_detect = time.time() - t0
    LOG(f"extract_faces found {len(found_faces)}  TOOK {t_detect:.3f}s")
    return found_faces


def main(argv, face_path = AFACE_PATH):
    args = argv[1:]
    for arg in args:
        if arg == "--aface_path":
            face_path = next(args)
            if not os.path.exists(face_path):
                os.makedirs(face_path)
            continue

        # Extract faces from file, align and save them to use as named faces for searching later
        extract_faces(arg, face_path)
        continue


if __name__ == "__main__":
    main(sys.argv)
