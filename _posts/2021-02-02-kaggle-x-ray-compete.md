---
layout: post
title: kaggle start ! Cheer up!
---
In this competition, you’ll detect the presence and position of catheters and lines on chest x-rays. Use machine learning to train and test your model on 40,000 images to categorize a tube that is poorly placed.

What files do I need?

You will need the train and test images. This is a code-only competition so there is a hidden test set (approximately 4x larger, with ~14k images) as well.

train.csv contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test. (They are also available for the hidden test set.)

We've also included train_annotations.csv. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

# Files

* train.csv - contains image IDs, binary labels, and patient IDs.
* sample_submission.csv - a sample submission file in the correct format
* test - test images
* train - training images

# Columns

StudyInstanceUID - unique ID for each image
* ETT - Abnormal - endotracheal tube placement abnormal
* ETT - Borderline - endotracheal tube placement borderline abnormal
* ETT - Normal - endotracheal tube placement normal
* NGT - Abnormal - nasogastric tube placement abnormal
* NGT - Borderline - nasogastric tube placement borderline abnormal
* NGT - Incompletely Imaged - nasogastric tube placement inconclusive due to imaging
* NGT - Normal - nasogastric tube placement borderline normal
* CVC - Abnormal - central venous catheter placement abnormal
* CVC - Borderline - central venous catheter placement borderline abnormal
* CVC - Normal - central venous catheter placement normal
* Swan Ganz Catheter Present
* PatientID - unique ID for each patient in the dataset