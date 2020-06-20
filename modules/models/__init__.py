def create_model(name, checkpoint_state, hparams):
    if name == 'LAS:
        from .las_clova import SpeechNetwork
        return SpeechNetwork(checkpoint_state, hparams)
    else:
        raise Exception('Unknown model: ' + name)
