# Learning To Classify Images Without Labels

<p>This repo contains the implementation of our paper _Learning To Classify Images Without Labels_ by Wouter Van Gansbeke<sup>\*</sup>, Simon Vandenhende<sup>\*</sup>, Stamatios Georgoulis, Marc Proesmans and Luc Van Gool.</p>

<p>Code + pretrained models + configuration files will be released (in a few weeks) to produce numbers even better than in the current version of the paper. </p>

<p>RotNet Implementation: https://github.com/d4nst/RotNet</p>


<p>Link to the Google Doc for notes: https://docs.google.com/document/d/1uJHiRPmBrf0Fj0HtmiOVGAohcRCq2iDiBanLasqcZd4/edit#</p>
 
<p><a href="https://arxiv.org/abs/2005.12320">Link to the Paper</a></p>
<p><a href="https://medium.com/@SeoJaeDuk/learning-to-classify-images-without-labels-43655a1cb4c7">Link to Tutorial by Jae Duk Seo</a></p>
<p><a href="https://arxiv.org/pdf/1805.01978.pdf">Link to Pretext Task Paper</a></p>

## Installing package

To install package gloabally run `python setup.py install`, to install in place use `python setup.py develop`

If you installed the package in a virtualenvironement use this command to add it to jupyter `python -m ipykernel install --user --name=<VENV_NAME>`

## TODO
- [x] Setup git repo
- [x] Setup package and dependencies
- [ ] Define APIs for Pretext/Clustering/self-labelling functions to easily extend
- [ ] Data augmentation utils:
    - [ ] Standard Data Augmentations function (use keras ImageGenerator directly?)
    - [x] Strong Data Augmentations function
    - [ ] Add cutout to strong data augmentation
    - [ ] Speedup Strong Data Augmentations function by using numpy func instead of PIL
- [ ] Implement utils:
    - [ ] NearestNeighbor datastruct
    - [ ] Resnet-18 backbone
- [ ] Pretext task (only one is strictly necessary):
    - [ ] RotNet (Ahsan)
    - [ ] Feature Decoupling
    - [ ] NCE
    - [ ] Training script
- [ ] Clustering task:
    - [ ] Custom loss
    - [ ] Training script
- [ ] Clustering results compared to true labels (Hungarian Algo)
- [ ] Demo jupyter notebook to get a result from the paper
