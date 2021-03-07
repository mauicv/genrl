def validate_model(model, *args, **kwargs):
    if not model.inputs:
        raise ValueError('Model has no input nodes')

    if not model.outputs:
        raise ValueError('Model has no output nodes')
