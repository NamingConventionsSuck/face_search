import sys
from pathlib import Path
import exiftool

EXIF_FIELDS = {
    'EXIF:DateTimeOriginal', 'EXIF:Make', 'EXIF:Model', 'EXIF:Software',
    'Composite:GPSDateTime', 'Composite:GPSPosition'
}

def extract_exif_for_path(filepath: str, img_glob: str = "*.jpg"):
    # TODO: exclude ._* files on Mac
    img_files = list(Path(filepath).rglob(img_glob, case_sensitive=False))
    print(len(img_files), f"image files in {filepath}")

    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(img_files)
        for md in metadata:
            fn = md["SourceFile"]
            md = {k:v for k,v in md.items() if k in EXIF_FIELDS}
            print(fn, "\t", md)


if __name__ == "__main__":
    argv = sys.argv
    img_path = argv[1] if len(argv) > 1 else "."
    extract_exif_for_path(img_path)
