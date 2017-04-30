import concurrent

import httplib2
import os
import argparse
import io

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient.http import MediaIoBaseDownload

SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Arvp Deep-learning Dataset'
FOLDER_IDS = {
    'root': '0BxfNkdUtSbXRSFp1cFVncWpiNFU',             # deep-learning
    'datasets': '0BxfNkdUtSbXRWDI3cmk0SzJXWW8',         # deep-learning/datasets
    'tagged_videos': '0BxfNkdUtSbXRcVpIZ0lZRzI5X3c',    # deep-learning/tagged_videos
    'raw_videos': '0BxfNkdUtSbXRU2M3dTY1NEJvbFk',       # deep-learning/raw_videos
    'image_db': '1AhC41ZzZrlNcrpuljxzoGBLm289_R1zZntM1yDei0Uk' # deep-learning/Image Database
}


def get_credentials(flags):
    """
    Gets valid user credentials from storage.
    If nothing has been stored, or if the stored credentials are incorrect, uses flow to obtain new auth credentials
    
    retrieved credentials are stored in ~/.credentials/arvp-dl-dataset.json
    
    :param flags: command line flags from argparser. Needed for run_flow function 
    :return: credentials
    """

    # setup path for credentials file
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, 'arvp-dl-dataset.json')

    # get credentials from stored file or use flow to retrieve new credentials
    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        credentials = tools.run_flow(flow, store, flags)
        print('Storing credentials to ' + credential_path)
    return credentials


def read_text_file(drive, id):
    request = drive.files().get_media(fileId=id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Downloaded {0}%".format(int(status.progress() * 100)))
    contents = fh.getvalue().decode("utf-8")
    return contents


def get_video_info(drive, video_name):
    query = "'{0}' in parents and name = '{1}.txt' and mimeType contains 'text'".format(
        FOLDER_IDS['raw_videos'], os.path.splitext(video_name)[0]
    )
    results = drive.files().list(q=query).execute()
    results = results.get('files', [])
    if len(results) != 1:
        print('Error retrieving txt file for {0}'.format(video_name))
        return None
    print("Reading file: {0}".format(results[0]['name']))
    info = {}
    for line in  read_text_file(drive, results[0]['id']).strip().split('\n'):
        key, value = line.split(':')
        info[key.strip()] = value.strip()
    return info


def get_list_of_videos(drive, read_info=False):
    query = "'{0}' in parents and mimeType contains 'video'".format(FOLDER_IDS['raw_videos'])
    results = drive.files().list(q=query).execute()
    items = results.get('files', [])
    if not items:
        print('No files found in folder. Ensure that the folder ID is correct and it still exists.')
        return None
    print('Files:')
    video_files = {}
    for item in items:
        video_files[item['name']] = {
            'id': item['id']
        }
        print(item['name'])
        if read_info:
            info = get_video_info(drive, item['name'])
            video_files[item['name']].update(info)
    return video_files


def export_video_info(drive, output_file, delimeter = ';'):
    videos = get_list_of_videos(drive, read_info=True)
    print('Writing video info to {0}'.format(output_file))
    with open(output_file, 'w') as ofile:
        ofile.write('Filename|Date|Camera|Pool|Orientation|Content|Conditions\n')
        for video, info in videos.items():
            ofile.write(video + '|')
            ofile.write(info['Date'] + '|')
            ofile.write(info['Camera'] + '|')
            ofile.write(info['Pool'] + '|')
            ofile.write(info['Orientation'] + '|')
            ofile.write(info['Content'] + '|')
            ofile.write(info['Conditions'] + '\n')


def main():
    # parse command-line flags
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()

    # authenticate and start gDrive service
    credentials = get_credentials(flags=flags)
    http = credentials.authorize(httplib2.Http())
    drive_service = discovery.build('drive', 'v3', http=http)

    # find deep learning folder id - set as global variable
    # root_folder_query = "name = '%s' and mimeType = '%s'" % (
    #     'deep_learning', 'application/vnd.google-apps.folder'
    # )
    # results = drive_service.files().list(q=root_folder_query).execute()
    # print( '%s: %s' % (results['files'][0]['name'], results['files'][0]['id']))

    # get csv file from raw video text files
    # export_video_info(drive_service, 'videos.csv')

if __name__ == '__main__':
    main()
