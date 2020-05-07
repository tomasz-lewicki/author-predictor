# CMPE 255 Data Mining Project


# Setup:
```shell
git clone https://github.com/tomek-l/cmpe255-data-mining.git
cd cmpe255-data-mining
conda create -f environment.yml
conda activate cmpe255
```

# To run notebooks:
```shell
conda activate cmpe255
ipython kernel install --user --name=255-data-mining
jupyter notebook 
```

# T. Lewicki's part:

## Data acqusition

To only view the "final product", please take look at ```utils/dataset.py```

Notebooks 1-2 and 4-8 illustrate the process of developing the data acquisiton pipeline and inferfacing with Gutenberg Index.

## Training
The training code is in ```09_tlewicki_final_model.ipynb```. 

GPU training requires ~```5.7GB``` of GPU memory. When running on a smaller GPU, please considering lowering ```BATCH_SIZE``` parameter, or running on CPU.

## Inference
The inference code is in ```10_tlewicki_inference.ipynb```

If you wish to run inference only (without prior training):

- Please download the [model files from google drive](https://drive.google.com/open?id=1q61poT6f5bQHGpT8xkt9SdZjYTpQsPeF)
- The file contains the following:
    - ```model.h5``` - keras NN model
    - ```tokenizer.json``` - word tokenizer
    - ```label_encoder.pkl``` - pickled label encoder
    - ```corpora.pkl``` - Pickled dataset to use in the event that online data acqustiton fails/is too long.

- Please unzip and place the files in project root directory. The code will find them and retreive the model.
