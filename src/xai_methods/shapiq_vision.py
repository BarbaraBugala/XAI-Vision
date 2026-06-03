from shapiq.vision import ImageImputer
from shapiq.vision.players import SuperpixelStrategy
from shapiq.vision.masking import MeanColorMasking, ZeroMasking # more masking is to be merged soon
import shapiq


def shapiq_vision(model, input_tensor, target, baseline=None, segments=None):
    input_image = input_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    # pass custom slic masks like this: player_strategy = SuperpixelStrategy(mask=segments)
    # ImageImputer(model, input_image, player_strategy=player_strategy)
    # Masking can be defined by using masking_strategy argument
    imputer = ImageImputer(model, input_image)
    
    approximator = shapiq.SHAPIQ(n=imputer.n_players, max_order=2, index="k-SII")
    interaction_values = approximator.approximate(budget=64, game=imputer)
    
    return interaction_values.interactions # This is probably not the final format you want, but gives the interaction values
