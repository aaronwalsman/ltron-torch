import os
import configparser

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Invalid Config Argument: %s'%key
            setattr(self, key, value)
        self.set_dependent_variables()
        
        self.kwargs = kwargs
    
    def set_dependent_variables(self):
        pass
    
    @classmethod
    def load_config(cls, file_path, section='CONFIG'):
        args = {}
        config = configparser.ConfigParser()
        config.read_file(open(os.path.expanduser(file_path)))
        for name in config[section]:
            try:
                args[name] = config[section].getint(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = config[section].getboolean(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = config[section].getfloat(name)
                continue
            except ValueError:
                pass
            
            args[name] = config[section][name]
        
        return cls(**args)
    
    def write_config(self, file_path, section='CONFIG'):
        config = configparser.ConfigParser()
        try:
            config.read_file(open(os.path.expanduser(file_path)))
        except FileNotFoundError:
            pass
        config[section] = self.kwargs
