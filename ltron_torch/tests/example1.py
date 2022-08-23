from ltron.config import Config
class MyTrainingConfig(Config):
    learning_rate = 3e-4
    momentum = 0.9

config = MyTrainingConfig.from_commandline()
print('learning_rate:', config.learning_rate)
print('momentum:', config.momentum)
