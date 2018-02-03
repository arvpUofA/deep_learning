"""
Gets info for all videos in raw_videos folder
Usage: python get_videos_list <output_csv>
    if output csv file given, then output goes to csv instead of std::out
"""

import sys
from drive_interface import DriveInterface


def main():
    output_file = None
    if len(sys.argv) == 2:
        output_file = sys.argv[1]

    drive = DriveInterface()
    videos = drive.get_list_of_videos(read_info=True)

    if output_file is None:
        # output to screen
        for video, info in videos.items():
            print('{}:'.format(video))
            print('\tdate: {}'.format(info['Date']))
            print('\tcamera: {}'.format(info['Camera']))
            print('\tpool: {}'.format(info['Pool']))
            print('\torientation: {}'.format(info['Orientation']))
            print('\tcontent: {}'.format(info['Content']))
            print('\tconditions: {}'.format(info['Conditions']))
            print('---------')
    else:
        # output to csv
        print('Writing video info to {}'.format(output_file))
        with open(output_file, 'w') as ofile:
            ofile.write('Filename;Date;Camera;Pool;Orientation;Content;Conditions\n')
            for video, info in videos.items():
                ofile.write(video + ';')
                ofile.write(info['Date'] + ';')
                ofile.write(info['Camera'] + ';')
                ofile.write(info['Pool'] + ';')
                ofile.write(info['Orientation'] + ';')
                ofile.write(info['Content'] + ';')
                ofile.write(info['Conditions'] + '\n')

if __name__ == '__main__':
    main()
