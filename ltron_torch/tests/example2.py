from ltron.config import Config
class MyTrainingConfig(Config):
    learning_rate = 3e-4
    momentum = 0.9

class MyNetworkConfig(Config):
    image_height = 224
    image_width = 224
    
    def set_dependents(self):
        super().set_dependents()
        self.num_pixels = self.image_height * self.image_width

class MyScriptConfig(MyTrainingConfig, MyNetworkConfig):
    pass

config = MyScriptConfig.from_commandline()
print('learning_rate:', config.learning_rate)
print('momentum:', config.momentum)
print('image_height:', config.image_height)
print('image_width:', config.image_width)
print('num_pixels:', config.num_pixels)
