def build_model(opts):
    
    if opts['model_type'] == 'GAN':
        from trainer.MODEL_gan import GANMODEL
        model = GANMODEL(opts)
    elif opts['model_type'] == 'DIFFUSION_GAN':
        from trainer.MODEL_gan_diffusion import DIFFGANMODEL
        model = DIFFGANMODEL(opts)
    else:
        from trainer.MODEL import MODEL
        model = MODEL(opts)
