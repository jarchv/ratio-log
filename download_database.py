import requests
import os
import zipfile

from sys import stdout

ID = '19tXbB3qG0laZT-kvibQwiLcbFdqavJcy'
DEST = 'FAIDres.zip'

def confirm_token(resp):
    for key, value in resp.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save(dest, resp, id, curr_size):
    def barsize(num):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}      '.format(num,unit,'B')
            num /= 1024.0
        return '{:.1f} {}{}'.format(num,'Yi',suffix)

    CHUNK_SIZE = 32768
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                print('\rDownloading '+id+' ... '+barsize(curr_size[0]), end=' ')
                stdout.flush()
                curr_size[0]+=CHUNK_SIZE
               
def download_drive_file(id, dest):
    if not os.path.exists('FAIDres.zip'):
        URL   = 'https://docs.google.com/uc?export=download'
        sess  = requests.Session()

        resp  = sess.get(URL, params = {'id': id}, stream = True)
        token = confirm_token(resp)

        if token:
            params = {'id':id, 'confirm':token}
            resp   = sess.get(URL, params=params, stream = True)
        
        curr_download_size = [0]
        save(dest, resp, id, curr_download_size)
        print('Done.')

    else:
        dest = "FAIDres.zip"
        
    if not os.path.exists('datasets/'):
        os.makedirs('datasets/')

    try:
        print('Unzipping...', end='')
        stdout.flush()

        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall('datasets')
        print('Done.')
    except zipfile.BadZipfile:
        print('Can not unzip file...')

if __name__ == '__main__':
    download_drive_file(ID,DEST)
