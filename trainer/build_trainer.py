
def build_trainer(opts):
    if opts['trainer'] == 'sgd':
        from trainer.Train import Trainer
        trainer = Trainer(opts)
    elif opts['trainer'] == 'adam':
        from trainer.Train_gan_diff import Trainer_gan_diff
        trainer = Trainer_gan_diff(opts)
    else:
        raise ValueError(f"Invalid trainer type: {opts['trainer']}")
