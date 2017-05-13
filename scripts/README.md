# Deep learning dataset scripts

## make_images.sh

Requires ffmpeg. Creates a folder of the same name as the video and converts video to images with proper naming

Usage: `./make_images.sh path_to_video`

## get_video_list.py

Returns a list of videos on drive. If a output file is specified, results are written to file, otherwise results are written to std::out.

Usage: `python get_videos_list <output_csv>`

## make_json_dataset.py

Creates a json format dataset based on the object type specified. 

Usage: `python make_json_dataset.py settings_file output_path`

If this is being run the first time, and credentials don't exist, follow the instructions, and sign in with your arvp gmail drive account to get autheticated.

## browse_dataset.py 

Allows viewing dataset images with ROIs drawn on them using the json file.

Usage: `python browse_dataset.py json_file_path dataset_path`

### Dependencies

* Python 3
* `pip install google-api-python-client`
* `pip install tqdm`
* `pip install pyyaml`
* `pip install pillow`
* opencv - `conda install -c https://conda.binstar.org/menpo opencv`