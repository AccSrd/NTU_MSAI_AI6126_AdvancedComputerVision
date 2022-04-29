#### [Basic Information]

The goal of this mini-challenge is to generate high-quality (HQ) face images from the corrupted low-quality (LQ) ones. The data for this task comes from the FFHQ. For this challenge, we provide a mini-dataset, which consists of 4000 HQ images for training and 400 LQ-HQ image pairs for validation.
The performance of the network will be evaluated based on the PSNR between the HR image and the ground truth of the test set. The website of this project can be seen [here](https://codalab.lisn.upsaclay.fr/competitions/2848).

This is the project of Advanced Computer Vision of MSAI, NTU.
We do not provide the tarining set, validation set and test set here. In this task, we use the FFHQ dataset. If you want, you can download the whole [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)

#### [Model Checkpoint]

The final checkpoint we use is the iter_2750000.pth of a MSRResNet-20 model.
You can access the checkpoint through the following link:...
*Here in Github, we do not make the checkpoint file public.

#### [Result Reproduction]

To reshow the final results, you need following files:
    1. mmediting enviornment
    2. The config file of testing, which is ./Code/train_configs/Final Model/FinalConfig.py
    3. The checkpoint file, which is ./Code/iter_2750000.pth
    4. Test images, which should be saved in mmediting's path: mmediting/data/test/LQ

Then, you can use following command to get the same results:
python tools/test.py ${config-path} ${checkpoint-path} --save-path ${save-path}

#### [Description of the files]

- Report.pdf -> the short report

- Code
  - mmediting
    - ...... -> The files which you need if you want to run the config files. Contain the dataset configs. If you want to use this, you can directly copy it to the root path of your mmsegmentation.
  - training configs
        - Final Model -> The training config file and json log of the final model.
        - others -> Other model we have tested, including configs which can run the model and the result json file.
    - train_log.sh -> The log of training process (Partly)
    - test_log.sh -> The log of testing process (Partly)

- PSNR Results.xlsx -> The table of all the testing PSNR results of this project.

#### [References]

[1] mmediting -> <https://github.com/open-mmlab/mmediting>
