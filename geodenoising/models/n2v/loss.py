from torch import Tensor


def loss_mask_mse(masked_coords: Tensor, input_tensor: Tensor, target: Tensor):
    mse = 0
    coords = masked_coords.tolist()[0]
    for coord in coords:
        x, y = coord
        diff = target[:, :, x, y] - input_tensor[:, :, x, y]
        mse += diff**2
    return mse
