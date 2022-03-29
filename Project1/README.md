#### [Basic Information]

Face parsing assigns pixel-wise labels for each semantic components, e.g., eyes, nose, mouth. The goal of this mini challenge is to design and train a face parsing network. We will use the data from the CelebAMask-HQ Dataset. For this challenge, we prepared a mini-dataset, which consists of 5000 training and 1000 validation pairs of images, where both images and annotations have a resolution of 512 x 512.
The performance of the network will be evaluated based on the mIoU between the
predicted masks and the ground truth of the test set. The website of this project can be seen [here](https://codalab.lisn.upsaclay.fr/competitions/1945?secret_key=a47e8cdb-3ad0-45cc-b182-13fa9ae81616).

This is the project of Advanced Computer Vision of MSAI, NTU.
We do not provide the tarining set, validation set and test set here. In this task, we use the mini-CelebA dataset. If you want, you can download the whole [CelebAMask-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ)

#### [Model Checkpoint]

The final checkpoint we use is the iter_72000.pth of a convnext model.
You can access the checkpoint through the following link:...
*Here in Github, we do not make the checkpoint file public.

#### [Result Reproduction]

To reshow the final results, you need following files:
    1. mmsegmentation enviornment
    2. The config file of testing, which is ./Code/training configs/TTA/final_convnext_l_test_TTA_4.py
    3. The checkpoint file, which can be downloaded through the link above
    4. Test images, which should be saved in mmsegmentation's path: /mmsegmentation/data/CelebA/image/test/

Then, you can use following command to get the same results:
python ./tools/test.py {config.py} {checkpoint.pth} --format-only --eval-options "imgfile_prefix={output_path}"

#### [Description of the files]

- Report.pdf -> the short report

- Code
  - mmsegmentation -> The files which you need if you want to run the config files. Contain the dataset configs. If you want to use this, you can directly copy it to the root path of your mmsegmentation. The images, masks, and pretrained checkpoints are deleted, only path remain.
  - scripts
    - fill_color.py ->This script is to fill the corresponding color in PALETTE to the mask.png output by ./tools/test.py -- format only command.
    - flip_celebA.py -> This script is to obtain the flip dataset mentioned in the report.
    - get_mean.py -> This script is to calculate the mean and std of the image.
  - training configs
    - final -> The training config file and json log of the final ConvNeXt-XL model.
    - baseline -> A series of baseline experiments' config code and json files.
    - others -> Other big model we have tested. Just configs which can run the model, without valid json files and results. Because we haedly fininsh the whole traning procedure of the test big model.
    - TTA -> The test-time augmentation config files, the number of them are corresponding to the report's Table.
  - train_log.sh -> The log of training process (Partly)
  - test_log.sh -> The log of testing process (Partly)

- Results
  - final_log_1/2.json -> The log json file of the final training process.

#### [References]

[1] mmsegmentation -> <https://github.com/open-mmlab/mmsegmentation>
