# Deep learning dataset scripts

## make_images.sh

Requires ffmpeg. Creates a folder of the same name as the video and converts video to images with proper naming

Usage: `./make_images.sh path_to_video`

## get_video_list.py

Returns a list of videos on drive. If a output file is specified, results are written to file, otherwise results are written to std::out.

Usage: `python get_videos_list <output_csv>`

## make_json_dataset.py

Creates a json format dataset based on the object type specified. 

Usage: `python make_json_dataset.py`

### Dependencies

* Python 3
* `pip install google-api-python-client`
* `pip install tqdm`