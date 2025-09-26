import yaml
import os

class Config:
    def __init__(self):

        def get_dir_name():
            dir_name = os.path.dirname(os.path.abspath(__file__))  
            #parent_dir = os.path.dirname(current_dir)    
            #current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
            return dir_name
        
        dir_name = get_dir_name()
        file_path = os.path.join(dir_name, "facts.yaml")
        
        with open(file_path, "r") as f:
            self.config = yaml.safe_load(f)

    @property
    def conditions(self):
        return [condition['description'] for condition in self.config['CONDITIONS']]
    @property
    def resolutions(self):
        return {condition['description']: condition['resolution'] for condition in self.config['CONDITIONS']}
    @property
    def happenings(self):
        return self.config["HAPPENINGS"]

    def get_resolution_for_condition(self, condition):
        return self.resolutions[condition]
