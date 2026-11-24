# geodenoising

This repository will contain tools for applying denoising tools in geophysical
applications, firstly applied to denoising LIDAR observations of water vapour
concentration below cloud in Earth's atmosphere.


## Supported models

| Model | Input | Description |
| --- | --- | --- |
| Noise2Clean | (x_noisy, x_clean) | Requires clean samples |
| Noise2Noise | (x_noisy1, x_noisy2) | Requires pairs of noisy samples with uncorrelated noise |
| Noise2Void | (x_noisy, x_noisy_manipulated) | Requires only noisy samples |
