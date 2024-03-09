from src.models.model_static_frame import get_or_create_model_static_frame


def get_model(model_type, input_shape, num_classes, seed=42):
    if model_type == 'static_frame':
        return get_or_create_model_static_frame(input_shape, num_classes, seed)
    else:
        raise Exception("Model not implemented")