# Deep learning dataset scripts

## make_images.sh

Requires ffmpeg. Creates a folder of the same name as the video and converts video to images with proper naming

Usage: `./make_images.sh path_to_video`

### Dependencies

* Python 3
* `pip install google-api-python-client`
* `pip install tqdm`
* `pip install pyyaml`
* `pip install pillow`
* opencv - `conda install -c https://conda.binstar.org/menpo opencv`
