"""
Google drive interface for arvp deep-learning data processing
"""

import os
import httplib2
import io
import argparse
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient.http import MediaIoBaseDownload

# readonly scope for GDrive
SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
# directory stores saved authenticated credentials
CREDENTIAL_DIR = os.path.join(os.path.expanduser('~'), '.credentials')
# application name
APPLICATION_NAME = 'Arvp Deep-learning Dataset'
# folder IDs for various folders
FOLDER_IDS = {
    'root': '0BxfNkdUtSbXRSFp1cFVncWpiNFU',             # deep-learning
    'datasets': '0BxfNkdUtSbXRWDI3cmk0SzJXWW8',         # deep-learning/datasets
    'tagged_videos': '0BxfNkdUtSbXRcVpIZ0lZRzI5X3c',    # deep-learning/tagged_videos
    'raw_videos': '0BxfNkdUtSbXRU2M3dTY1NEJvbFk',       # deep-learning/raw_videos
    'image_db': '1AhC41ZzZrlNcrpuljxzoGBLm289_R1zZntM1yDei0Uk' # deep-learning/Image Database
}


class DriveInterface(object):
    """
    Creates a drive connection, and fetches various files
    """
    def __init__(self, secrets_file='./client_secret.json'):
        flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
        credentials = DriveInterface.get_credentials(secrets_file=secrets_file, flags=flags)
        http = credentials.authorize(httplib2.Http())
        self.drive_service = discovery.build('drive', 'v3', http=http)

    def get_list_of_videos(self, read_info=False):
        """
        gets list of videos (and their info) inside raw_videos
        :param read_info: flag to read video info
        :return: 
        """
        # query = "'{0}' in parents and mimeType contains 'video'".format(FOLDER_IDS['raw_videos'])
        # results = self.drive_service.files().list(q=query).execute()
        # items = results.get('files', [])
        items = self.get_files(
            mime_type='video',
            parents=FOLDER_IDS['raw_videos']
        )
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
                info = self.get_video_info(item['name'])
                video_files[item['name']].update(info)
        return video_files

    def get_video_info(self, video_name):
        """
        gets info for video file in raw_videos by reading the text file associated with video
        :param video_name: name of video file
        :return: dictionary with video information
        """
        # query = "'{0}' in parents and name = '{1}.txt' and mimeType contains 'text'".format(
        #     FOLDER_IDS['raw_videos'], os.path.splitext(video_name)[0]
        # )
        # results = self.drive_service.files().list(q=query).execute()
        # results = results.get('files', [])
        results = self.get_files(
            file_name=os.path.splitext(video_name)[0] + '.txt',
            mime_type='text',
            parents=FOLDER_IDS['raw_videos']
        )
        if len(results) != 1:
            print('Error retrieving txt file for {0}'.format(video_name))
            return None
        print("Reading file: {0}".format(results[0]['name']))
        info = {}
        for line in self.read_text_file(results[0]['id']).strip().split('\n'):
            key, value = line.split(':')
            info[key.strip()] = value.strip()
        return info

    def get_folder_id_by_name(self, folder_name):
        """
        get google drive id for specified folder
        :param folder_name: name of folder
        :return: google drive folder id
        """
        results = self.get_files(
            file_name=folder_name,
            mime_type='application/vnd.google-apps.folder'
        )
        if len(results) == 0:
            print('No folders with name {0} found'.format(folder_name))
            return None
        elif len(results) > 1:
            print('{0} folder with name {1} found'.format(len(results), folder_name))
            for folder in results:
                print('{0}: {1}'.format(folder['name'], folder['id']))
            return None
        else:
            return results[0]['id']

    def get_files(self, file_name=None, mime_type=None, parents=None):
        """
        get list of files
        :param file_name: name of file 
        :param mime_type: file mime type
        :param parents: instance id of parent folder
        :return: list of files
        """
        query = ""
        if file_name:
            query += "name = '{}'".format(file_name)
        if mime_type:
            if file_name:
                query += " and "
            query += "mimeType contains '{}'".format(mime_type)
        if parents:
            if mime_type or file_name:
                query += " and "
            query += "'{}' in parents".format(parents)
        results = self.drive_service.files().list(q=query).execute()
        return results.get('files', [])

    def download_file(self, file_id, output_file, progress=False):
        """
        downloads file to disk
        :param file_id: google drive file idd
        :param output_file: local destination on disk
        :param progress: flag to show progress print statements
        """
        contents = self.get_file_contents(file_id, progress)
        with open(output_file, 'wb') as o_file:
            o_file.write(contents)

    def read_text_file(self, file_id, progress=False):
        """
        reads a text file from drive to memory
        :param file_id: google drive file id
        :param progress: flag to show progress print statements
        :return: string contents of file
        """
        contents = self.get_file_contents(file_id, progress).decode("utf-8")
        return contents

    def get_file_contents(self, file_id, progress):
        """
        reads contents of a file to memory
        :param file_id: google drive file id
        :param progress: flag to show progress print statements
        :return: contents of file
        """
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if progress:
                print("Downloaded {0}%".format(int(status.progress() * 100)))
        return fh.getvalue()

    @staticmethod
    def get_credentials(secrets_file, flags):
        """
        Gets valid user credentials from storage.
        If nothing has been stored, or if the stored credentials are incorrect, uses flow to obtain new auth credentials
    
        retrieved credentials are stored in CREDENTIAL_DIR/arvp-dl-dataset.json
        :param secrets_file: file containing client secrets
        :param flags: command line flags from argparser. Needed for run_flow function 
        :return: authenticated credentials
        """
        # setup path for credentials file
        if not os.path.exists(CREDENTIAL_DIR):
            os.makedirs(CREDENTIAL_DIR)
        credential_path = os.path.join(CREDENTIAL_DIR, 'arvp-dl-dataset.json')

        # get credentials from stored file or use flow to retrieve new credentials
        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(secrets_file, SCOPES)
            flow.user_agent = APPLICATION_NAME
            credentials = tools.run_flow(flow, store, flags)
            print('Storing credentials to ' + credential_path)
        return credentials
