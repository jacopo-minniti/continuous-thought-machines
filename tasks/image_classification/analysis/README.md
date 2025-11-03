# Analysis

This folder contains the analysis code for the image classifcation experiments. Running the following from the base directory will generate figures, gifs and mp4 files:

```
python -m tasks.image_classification.analysis.run_imagenet_analysis
```

To cache ImageNet mistakes (predictions and confidence traces) run:

```
python -m tasks.image_classification.analysis.run_imagenet_mistake_analysis --checkpoint <imagenet_checkpoint>
```
