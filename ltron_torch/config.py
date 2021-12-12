import os
import configparser

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Invalid Config Argument: %s'%key
            setattr(self, key, value)
        
        for Class in self.__class__.mro():
            if isinstance(Class, Config):
                super(Class, self).set_dependents()
        
        self.kwargs = kwargs
    
    def set_dependents(self):
        pass
    
    @classmethod
    def load_config(cls, file_path, section='CONFIG'):
        args = {}
        parser = configparser.ConfigParser()
        parser.read_file(open(os.path.expanduser(file_path)))
        for name in parser[section]:
            try:
                args[name] = parser[section].getint(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getboolean(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getfloat(name)
                continue
            except ValueError:
                pass
            
            value_string = parser[section][name]
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
        parser = configparser.ConfigParser()
        try:
            parser.read_file(open(file_path))
        except FileNotFoundError:
            pass
        parser[section] = self.kwargs
        with open(file_path, 'w') as f:
            parser.write(f)

class OldConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Invalid Config Argument: %s'%key
            setattr(self, key, value)
        self.set_dependents()
        self.kwargs = kwargs
    
    def set_dependents(self):
        pass
    
    @classmethod
    def load_config(cls, file_path, section='CONFIG'):
        args = {}
        parser = configparser.ConfigParser()
        parser.read_file(open(os.path.expanduser(file_path)))
        for name in parser[section]:
            try:
                args[name] = parser[section].getint(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getboolean(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getfloat(name)
                continue
            except ValueError:
                pass
            
            value_string = parser[section][name]
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
        parser = configparser.ConfigParser()
        try:
            parser.read_file(open(file_path))
        except FileNotFoundError:
            pass
        parser[section] = self.kwargs
        with open(file_path, 'w') as f:
            parser.write(f)

class CompositeConfig(Config):
    MemberClasses = ()
    def __init__(self, **kwargs):
        member_args = [{} for MemberClass in self.MemberClasses]
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                found_key = True
            else:
                found_key = False
            for args, MemberClass in zip(member_args, self.MemberClasses):
                if hasattr(MemberClass, key):
                    args[key] = value
                    found_key = True
            assert found_key, 'Invalid Config Argument: %s'%key
        self.set_dependents()
        self.kwargs = kwargs
        

def CompositeConfig(ConfigClasses):
    class CustomCompositeConfig(Config):
        def __init__(self, **kwargs):
            #self.configs = [ConfigClass() for ConfigClass in ConfigClasses]
            config_args = [{} for ConfigClass in ConfigClasses]
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    found_key = True
                else:
                    found_key = False
                for ConfigClass in ConfigClasses:
                    if hasattr(config, key):
                        setattr(config, key, value)
                        found_key = True
                assert found_key, 'Invalid Config Argument: %s'%key
            self.set_dependents()
            self.kwargs = kwargs
        
        def __getattr__(self, attr):
            for config in self.configs:
                if hasattr(config, attr):
                    return getattr(config, attr)
            
            raise AttributeError(attr)
        
        def __setattr__(self, attr, value):
            for config in self.configs:
                if hasattr(config, attr):
                    setattr(config, attr, value)
            
            raise AttributeError(attr)
        
        def set_dependents(self):
            for config in self.configs:
                config.set_dependents()
    
    return CustomCompositeConfig
