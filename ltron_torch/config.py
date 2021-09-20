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
            
            value_string = config[section][name]
            if ',' in value_string:
                for remove in '[]()':
                    value_string = value_string.replace(remove, '')
                def convert_value(value):
                    try:
                        v = float(value)
                        if v.is_integer():
                            return int(v)
                        else:
                            return v
                    except ValueError:
                        return value
                values = tuple(
                    convert_value(v) for v in value_string.split(','))
                args[name] = values
                continue
            
            args[name] = value_string
        
        return cls(**args)
    
    def write_config(self, file_path, section='CONFIG'):
        file_path = os.path.expanduser(file_path)
        config = configparser.ConfigParser()
        try:
            config.read_file(open(file_path))
        except FileNotFoundError:
            pass
        config[section] = self.kwargs
        print('look at me go!')
        with open(file_path, 'w') as f:
            config.write(f)
