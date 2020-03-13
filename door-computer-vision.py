from os import getcwd, listdir
from os.path import isfile, join, splitext
from wget import download

from imageai.Detection import ObjectDetection

project_path = getcwd()
source_path = join(project_path, "source")
target_path = join(project_path, "target")
dataset_path = join(project_path, "resnet50_coco_best_v2.0.1.h5")
images_files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and splitext(f)[1][1:] == "jpg"]

if not isfile(dataset_path):
    def bar_custom(current, total, width=80):
        print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))
    download(
        "https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5",
        dataset_path,
        bar=bar_custom
    )

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(dataset_path)
detector.loadModel()

for image in images_files:
    detections = detector.detectObjectsFromImage(
        input_image=join(source_path, image),
        output_image_path=join(target_path, image),
        extract_detected_objects=True,
        minimum_percentage_probability=65
    )
    for objects in detections:
        if len(objects) > 0 and isinstance(objects[0], dict):
            for obj in objects:
                print(image, obj["name"], obj["percentage_probability"])
