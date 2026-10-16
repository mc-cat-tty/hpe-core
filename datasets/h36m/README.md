# Event-Human 3.6 million (quick and suggested way)
[Event-Human 3.6m million](https://zenodo.org/records/7842598) is a dataset containing a conversion of Human 3.6m dataset RGB videos to event streams.

### Usage
- Install dependencies by running `pip install -r datasets/h36m/requirements.txt`, possibly in a virtual environment.
- Download the dataset by running `bash datasets/h36m/download_eh36m.sh <EH36M-FOLDER>`
- Run `python -m datasets.h36m.h36_h5_to_yarp --input_folder <EH36M-FOLDER> --output_folder <EH36M-FOLDER> --h5_name data.h5` to convert the output of `v2e` to the YARP format.
- Run `python -m datasets.h36m.h36m_to_eros --input_folder <EH36M-FOLDER> --output_folder <EH36M-FOLDER> --write_images --write_annotation --from_scratch` to convert the events to the EROS representation.

# Original Human 3.6 million to events

[Human 3.6m](http://vision.imar.ro/human3.6m/description.php) is an image-based dataset recorded at 50Hz. Event-based dataset can be created using [v2e](https://github.com/SensorsINI/v2e).

### Usage
- Install dependencies by running `pip install -r datasets/h36m/requirements.txt`, possibly in a virtual environment.
- Download [Human 3.6m](http://vision.imar.ro/human3.6m/description.php) dataset.
- Run `python -m datasets.h36m.h36m_v2e --dataset_path <DATASET-PATH> --output_folder <OUT-FOLDER>` to create high temporal resolution events from frame inputs. Cropping can be applied to get a video converted to `640x480` centered on the person in the image.
- `python -m datasets.h36m.h36_h5_to_yarp --input_folder <INPUT-FOLDER> --output_folder <OUT-FOLDER> --h5_name data.h5` is used to convert the output of `v2e` to the YARP format.
- `python -m datasets.h36m.h36_gt_frames_to_yarp.py` is used to convert the ground-truth and images to YARP format. If cropping is used in step 1, the same cropping can be used to modify the images and ground-truth joint positions.
- `h36m_to_xxx.py` is used to convert YARP format events to image-based formats used for training and testing detectors. To convert the events to the EROS representation run `python -m datasets.h36m.h36m_to_eros --input_folder <EH36M-FOLDER> --output_folder <EH36M-FOLDER> --write_images --write_annotation --from_scratch`