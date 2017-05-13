if [ "$#" -ne 1 ]; then
	echo "Usage: ./make_images.sh path_to_video"
fi

VIDEO_PATH=$1
VIDEO_NAME="$(basename $VIDEO_PATH)"
FOLDER_NAME="${VIDEO_NAME%*.*}"

echo "Making folder $FOLDER_NAME"
mkdir $FOLDER_NAME

ffmpeg -i $VIDEO_PATH -r 5/1 $FOLDER_NAME/$filename$FOLDER_NAME-%05d.jpg
