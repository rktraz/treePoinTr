# -*- coding: utf-8 -*-

# write .txt files (train.txt, test.txt) with list of filenames to use for either training or testing

import functions

# define: path to full shape clouds, path to partial shape clouds, output path for .txt files, filenames for test and train .txt files
functions.train_test_txt_predef("./data/predefhull/complete/", "./data/predefhull/partial/", "./data/predefhull/", txt1= "test13.txt", txt2 = "train13.txt")
print("done")

