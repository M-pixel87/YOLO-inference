from roboflow import Roboflow
rf = Roboflow(api_key="eJplbmcLOrszEMR1DcgZ") 
project = rf.workspace("avc20252026").project("yolo_inferencing-e1txa")
version = project.version(1)
dataset = version.download("yolov8")
                
