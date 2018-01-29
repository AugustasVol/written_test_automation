#!/usr/bin/env python
from written_test_automation import pre, result
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)    
def show(im):
    plt.imshow(im,cmap="gray")
    plt.show()
import fire
def im_predict(path, visual=False, visual_net=False, save_files=False):
    im = pre.imread(path)
    #show(im)
    print(np.argmax(result.predict(im, visual=visual, save_files=save_files, visual_net=visual_net), axis = 1))
                                                                                
fire.Fire(im_predict)
