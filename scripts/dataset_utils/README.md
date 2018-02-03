## make_json_dataset.py

Creates a json format dataset based on the object type specified. 

Usage: `python make_json_dataset.py settings_file output_path`

If this is being run the first time, and credentials don't exist, follow the instructions, and sign in with your arvp gmail drive account to get autheticated.


## json_to_kitti.py

Converts json dataset to kitti dataset format.

Usage: `python json_to_kitti.py input_json output_folder object`

## json_to_yolo.py

Converts json dataset to yolo dataset format.

Usage: `python json_to_yolo.py input_json output_folder object_name image_height image_width names_file



## make_local_dataset.py

Creates yolo dataset from downloaded tagged_images folder (eg. https://drive.google.com/drive/u/0/folders/1-Wal99FcjnkGAZfNsXmjOW-mV80iHbCP)

Usage: `python make_local_dataset.py settings_file output_path tagged_images_dir`
  
Help: `python make_local_dataset.py -h`

## make_yolo_dataset.py
