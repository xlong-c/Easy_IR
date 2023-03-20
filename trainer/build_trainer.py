
def build_trainer(opts):
    if opts['trainer'] == 'trainer':
        from trainer.Train import Trainer
        trainer = Trainer(opts)
    elif opts['trainer'] == 'trainer_gan_diff':
        from trainer.Train_gan_diff import Trainer_gan_diff
        trainer = Trainer_gan_diff(opts)
    else:
        raise ValueError(f"Invalid trainer type: {opts['trainer']}")
    return trainer
