import dlib
import numpy as np

import os
import sys
import logging
import time
from pathlib import Path

PROG = sys.argv[0]
# A small number of known faces should be are saved
AFACE_PATH = "~/Pictures/known_faces"
AFACE_PREFIX = "aface_"

USAGE_MSG = f"""
Usage: {PROG}
        [--aface_path /path/to/aface files]       # default is {AFACE_PATH}
         --save_faces _face_glob_or_files_
        _image_glob_or_file_ ...

Face matching should be done in these steps:
1. Select a few photos containing the faces you want to tag
2. Run this script to extract faces from the selected photos above and save to aface path
   python {PROG} -aface_path {AFACE_PATH}  -save_faces  selected/photos*.jpg
3. View the *_aface_* files saved above, and select those you want to tag and rename:
    cd ~/Pictures/aface
    mv selected_aface_??.jpg  aface_NameOfPerson.jpg
4. Run this script again to match any image files to match against saved named faces
    python {PROG} -aface_path {AFACE_PATH}  path/to/image/files*.jpg
    python {PROG}  path/to/images/      # all images under this folder
"""


# Models for dlib face recognition. See installation instruction above
DLIB_FACE_DETECTOR_FILE = "mmod_human_face_detector.dat"     # <1MB
DLIB_SHAPE_MODEL_FILE = "shape_predictor_5_face_landmarks.dat"      # 9MB
# DLIB_SHAPE_MODEL_FILE = "models/shape_predictor_68_face_landmarks.dat"    # 95MB
DLIB_FACE_REC_MODEL_FILE = "dlib_face_recognition_resnet_model_v1.dat"      # 21MB

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "models")

# Load Dlib models for face detector, landmark predictor, and face recognition
# To chose what to use, watch https://youtu.be/j27xINvkMvM?t=2839
FACE_DETECTOR = dlib.get_frontal_face_detector()    # HOG detector
# CNN_FACE_DETECTOR = dlib.cnn_face_detection_model_v1(os.path.join(MODEL_PATH, DLIB_FACE_DETECTOR_FILE))
FACE_PREDICTOR = dlib.shape_predictor(os.path.join(MODEL_PATH, DLIB_SHAPE_MODEL_FILE))
FACE_ENCODER = dlib.face_recognition_model_v1(os.path.join(MODEL_PATH, DLIB_FACE_REC_MODEL_FILE))

COS_SIM = 'cosine'      # cosine similarity
EUCLID_L2 = 'l2'        # euclidean L2 distance
# Thresholds for face matching.  These are taken from DeepFace:
#  https://github.com/serengil/deepface/blob/master/deepface/modules/verification.py
DLIB_THRESHOLDS = {
    COS_SIM: 0.07,
    EUCLID_L2: 0.4,
}


logging.basicConfig(stream=sys.stderr, level=logging.INFO, datefmt="%H%M%S",
                    format='I%(asctime)s %(filename)s:%(lineno)d: %(message)s')
LOG = logging.info


def rgb_to_gray_img(rgb_img):
    """ Converts an RGB image to 8-bit grayscale (speedup face detection)
    """
    img = rgb_img
    gray = np.round(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    # return dlib.pyramid_down(dlib.array_to_image(gray))
    return gray

def halfsize_img(gray_img):
    new_img = gray_img[::2, ::2]
    #new_img = cv2.resize(gray_img, (gray_img.shape[1] // 2, gray_img.shape[0] // 2))    # interpolation=cv2.INTER_LINEAR)
    return new_img


def cos_similarity(av, bv):
    """ Compute cosine similarity (angle) between 2 embedding vectors av, bv """
    dot_prod = np.dot(av, bv)
    mag_a = np.linalg.norm(av)
    mag_b = np.linalg.norm(bv)
    return dot_prod / (mag_a * mag_b)

def l2_dist(av, bv):
    """ Compute Euclidean distance (L2) between 2 embedding vectors av, bv """
    return np.linalg.norm(av - bv, ord=2)


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


def load_named_faces(face_paths):
    """ Load known face files (with aface_ prefix) and put in a map from:
      key: the {name} substring of the aface_{name}.jpg
      val: the embedding vector computed from known face image
    """
    LOG(f"load_named_face from: {face_paths}")
    t0 = time.time()
    name_to_facev = dict()   # map from name string to list(embeddings)
    num_skip = 0
    last_ev = None
    last_fn = ""
    for ffile in face_paths:
        given_name = os.path.splitext(os.path.basename(ffile))[0]
        if given_name.startswith(AFACE_PREFIX):
            given_name = given_name[6:]
        rgb_img = dlib.load_rgb_image(ffile)
        rects = FACE_DETECTOR(rgb_img, 0)
        if len(rects) != 1:
            num_skip += 1
            continue

        shape = FACE_PREDICTOR(rgb_img, rects[0])
        face_chip = dlib.get_face_chip(rgb_img, shape)  # crop+align face
        ev = np.array(FACE_ENCODER.compute_face_descriptor(face_chip))
        # ev = np.array(FACE_ENCODER.compute_face_descriptor(rgb_img, face))
        name_to_facev[given_name] = ev

        if last_ev is not None:
            cs = cos_similarity(ev, last_ev)
            l2 = l2_dist(ev, last_ev)
            print(f"\tl2={l2:.3f} cos_sim={cs:.3f} between {given_name} and {last_fn}")
        last_fn = given_name
        last_ev = ev

    tt = time.time() - t0
    num = len(name_to_facev)
    LOG(f"name_to_face: took {tt:.3}s to load {num} afaces")
    return name_to_facev


def match_face_to_names(face_ev, name_to_facev, method=EUCLID_L2):
    match_names = dict()
    for name, known_ev in name_to_facev.items():
        if method == COS_SIM:
            delta = cos_similarity(face_ev, known_ev)
            if delta <= DLIB_THRESHOLDS[COS_SIM]:
                match_names[name] = float(delta)
        else:
            dist = l2_dist(face_ev, known_ev)
            if dist <= DLIB_THRESHOLDS[EUCLID_L2]:
                match_names[name] = float(dist)
    return match_names


def find_named_faces(image_paths, name_to_facev):
    LOG(f"find_named_face {image_paths} matching against {len(name_to_facev)} known faces")

    # img_to_names maps from the image file name to the names of the faces in the image
    img_to_names = dict()
    t_detect = t_predict = t_encode = t_match = 0

    tot_img = len(image_paths)
    n_no_face = 0
    GRAY_HALF = 1
    for img_fn in image_paths:
        bname = os.path.basename(img_fn)
        rgb_img = dlib.load_rgb_image(img_fn)
        gray_img = rgb_to_gray_img(rgb_img)
        if GRAY_HALF:  gray_img = halfsize_img(gray_img)

        t0 = time.time()
        det_faces = FACE_DETECTOR(gray_img, 0)  # upsample_num_times=0 runs faster
        t_detect += time.time() - t0
        if not det_faces:
            print(f" {bname}:\tNO_FACE")
            n_no_face += 1
            continue

        # Match any detected faces against all known faces loaded in named_to_facev earlier
        # through L2 or Cosine Similarity between 2 embedding vectors
        found_names = dict()
        for d in det_faces:
            rect = d
            if GRAY_HALF:  dlib.rectangle(d.left()*2, d.top()*2, d.right()*2, d.bottom()*2)
            t0 = time.time()
            shape = FACE_PREDICTOR(rgb_img, rect)

            # create a face chip (crop+align) and generate embedding vector, ev
            t1 = time.time()
            face_chip = dlib.get_face_chip(rgb_img, shape)  # crop+align face
            ev = np.array(FACE_ENCODER.compute_face_descriptor(face_chip))

            # Compare this detected face ev against the known named faces in named_to_facev
            t2 = time.time()
            match_names = match_face_to_names(ev, name_to_facev)
            if match_names:
                found_names.update(match_names)
                # LOG(f"\t{match_names} match in {bname}  {len(found_names)}")
            t_predict += t1 - t0
            t_encode += t2 - t1
            t_match += time.time() - t2

        if found_names:
            # Round match value to 3 decimal places
            found_names = {k: round(v,3) for k,v in found_names.items()}
            img_to_names[img_fn] = list(found_names.keys())
            print(f" {bname}:\t{found_names}")
        else:
            print(f" {bname}:\tNO_MATCH  {len(det_faces)} detected faces")

    # Print out a summary
    m_img = len(img_to_names)
    m_face = sum([len(v) for v in img_to_names.values()])
    has_face = len(image_paths) - n_no_face
    LOG(f"find_named_faces: {m_face} faces / {m_img} match_img/ {has_face} has_face/ {tot_img} total_img"
        f"  TOOK detect:{t_detect:.1f}s predict:{t_predict:.3f} encode:{t_encode:.3f} match:{t_match:.3f}")
    return img_to_names


def rglob_if_has_kw(path_glob: str, recursive_kw="/**"):
    if path_glob.startswith('~'):
        path_glob = os.path.expanduser(path_glob)
    if os.path.isdir(path_glob):
        fpath = path_glob
        fglob = '*.jpg'
    else:
        fpath = os.path.dirname(path_glob)
        fglob = os.path.basename(path_glob)
        pos = fpath.find(recursive_kw)
        if pos > 0:
            fpath = path_glob[:pos]
    # Recursive find files matching fglob under fpath
    files = sorted(str(p) for p in Path(fpath).rglob(fglob))   #, case_sensitive=False)]
    LOG(f"rglob: {len(files)} for {fpath}/ {fglob}")
    return files


def Usage(msg=""):
    print(USAGE_MSG)
    exit(msg)

def main(argv, face_path = AFACE_PATH):
    argc = len(argv)
    if argc <= 1:
        Usage("Please provide some files or glob (eg, '*.jpg') to match with named faces")

    args = argv[1:]
    cmd_mode = 'match'
    name_to_facev = {}
    for arg in args:
        if arg == "--aface_path":
            face_path = next(args)
            if not os.path.exists(face_path):
                os.makedirs(face_path)
            continue
        elif arg == "--save_face":
            cmd_mode = 'save_face'
            continue

        if cmd_mode == 'save_face':
            # Extract faces from file, align and save them to use as named faces for searching later
            extract_faces(arg, face_path)
            continue

        if not name_to_facev:
            # Load the known face files 'aface_*.jpg' in the face_path
            file_paths = rglob_if_has_kw(face_path + f"/**/{AFACE_PREFIX}*.jpg")
            name_to_facev = load_named_faces(file_paths)

        file_list = rglob_if_has_kw(arg)
        if not file_list:
            continue

        # For each image file provided on the commandline, extract the faces and match them
        # against known face (loaded to name_to_facev), and print all the names that match.
        find_named_faces(file_list, name_to_facev)

if __name__ == "__main__":
    main(sys.argv)
