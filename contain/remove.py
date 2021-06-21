import os
import time

path = os.path.dirname(os.path.realpath(__file__))
day = 1
seconds = time.time() - (day * 24 * 60 * 60)
deleted_files_count = 0


def get_file_or_folder_age(path):

    ctime = os.stat(path).st_ctime
    return ctime

def remove_file(path):
    os.remove(path)

def run_opp():
    real_path = os.path.join(path, "tempdir")
    if os.path.exists(real_path):
        for root_folders, folders, files in os.walk(real_path):
            for file in files:
                file_path = os.path.join(root_folders, file)
                if seconds >= get_file_or_folder_age(file_path):
                        remove_file(file_path)
                        
 
    

