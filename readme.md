# InstantDream
This is an implementation of [DreamFields](https://github.com/google-research/google-research/tree/master/dreamfields) with fast NeRF implementation ([InstantNGP](https://github.com/NVlabs/instant-ngp)). This repo is built of [torch-ngp](https://github.com/ashawkey/torch-ngp).

## Usage
Install dependencies using the provided `environment.yaml` file.
Run experiments following templates in `./scripts/run_dream.sh`. You can try both the `mlp` or the `hash` version.

## Potential Improvements
1. Use mip-nerf as in the original paper.
2. Tune the weights of differences including transmittance loss and origin loss.
3. Tune the scheduling of transmittance loss.
4. Change activation functions (currently relu).