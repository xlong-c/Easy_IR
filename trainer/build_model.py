def build_model(opts):
    if opts['model_type'] == 'GAN':
        """
        This is the model for gan
        有gan网络的判别器网络
        """
        from trainer.MODEL_gan import GANMODEL
        model = GANMODEL(opts)
    elif opts['model_type'] == 'DIFFUSION_GAN':
        """
        This is the model for diffusion gan
        """
        from trainer.MODEL_gan_diffusion import DIFFGANMODEL
        model = DIFFGANMODEL(opts) 
    else:
        from trainer.MODEL import MODEL
        model = MODEL(opts)
    return model
