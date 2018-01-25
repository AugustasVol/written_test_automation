#!/usr/bin/env python
from written_test_automation import pre, result
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)    
def show(im):
    plt.imshow(im,cmap="gray")
    plt.show()
import fire
def im_predict(path):
    im = pre.imread(path)
    show(im)
    print(result.predict(im, visual=True))
                                                                                
fire.Fire(im_predict)
