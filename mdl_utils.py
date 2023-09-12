import shutil

"""
ModelSyncher: store a locally updated model to cache in Google Drive. 

    Note 1. Best to load at start of NB
    Note 2. Save during updates in the NB
"""
class ModelSyncher:
    def __init__(self, drive_subfolder='data'):
       self.drive_folder = f"/content/drive/MyDrive/{drive_subfolder}"

    def start(self, model_path):
        self.model_path = model_path

    def update(self):   
        shutil.copy(self.model_path,  self.drive_folder + self.model_path)

    def stop(self):
        pass
