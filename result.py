#!/usr/bin/env python3

import loc
import pre
import numpy as np
import config
import nets
import matplotlib.pyplot as plt
from uuid import uuid4
def show(im):
    plt.imshow(im,cmap="gray")
    plt.show()


net = nets.answer_model()
net.load_weights("./answer_weights.pickle")

class fiver_iterator:
    def __init__(self, im, blob_loc):
        self.im = im
        self.blob_loc = blob_loc

        self.column = 0
        self.row = 0

        self.row_number = blob_loc.shape[1]
        self.column_number = blob_loc.shape[0]

        if self.column_number not in [2,4]:
            raise

    def __iter__(self):
        self.column = 0
        self.row = 0
        return self
    def __next__(self):

        if self.row >= self.row_number - 1:
            self.row = 0
            self.column += 2
            if self.column >= self.column_number - 1:
                raise StopIteration()


        column_one = 0 + self.column
        column_two = 1 + self.column
        row_one = self.row
        row_two = 1 + self.row
            
        one_one = self.blob_loc[column_one][row_one]
        two_one = self.blob_loc[column_two][row_one]
        one_two = self.blob_loc[column_one][row_two]
        two_two = self.blob_loc[column_two][row_two]
            
        dim11 = int( (one_one[0] + one_two[0]) / 2 )
        dim12 = int( (two_one[0] + two_two[0]) / 2 )
            
        dim21 = int( (one_one[1] + two_one[1]) / 2 )
        dim22 = int( (one_two[1] + two_two[1]) / 2 )
        #print(self.blob_loc[column_one][row_one], self.blob_loc[column_two][row_one])
        #print(self.blob_loc[column_one][row_two], self.blob_loc[column_two][row_two])
        #print()
            
        im_fiver = self.im[ dim21 : dim22 , dim11 : dim12 ]

        #print(self.row,self.column)

        self.row += 1

        return im_fiver

def predict(im, black_dots_columns = 4):

    ### calculate configs

    largest_dimension = np.max(im.shape)

    threshold_kernel = pre.odd_number(largest_dimension * config.threshold_kernel_percent)
    median_blur_kernel = pre.odd_number(largest_dimension * config.median_blur_kernel_percent)
    blob_min_area = int(3.14 * (((largest_dimension * config.blob_min_size_percent)/2)**2))

    ###

    ### initialize blob detector

    blob_detector = loc.blobs(min_area=blob_min_area)

    ###

    ### prepare images for blob detection

    im_th = pre.threshold(im, kernel_size=threshold_kernel, C = config.threshold_C)
    im_median = pre.median_blur(im_th, kernel_size=median_blur_kernel)
    im_median = pre.resize(im_median, (im_th.shape[1],im_th.shape[0]))

    ###
    #show(im)
    #show(im_th)
    #show(im_median)

    ### detect blobs from median_blur_image and get deskew angle

    blobs = blob_detector.blob_location(im_median)
    outer_4 = loc.locate_4_outer_points(blobs)
    outer_side = loc.sides_from_outer_points(outer_4)
    left_angle = loc.angle_2_points(outer_side["left"][0], outer_side["left"][1])
    right_angle = loc.angle_2_points(outer_side["right"][0],outer_side["right"][1])
    skew_angle = np.mean([left_angle, right_angle])

    ###

    #print(left_angle, right_angle)

    ### rotate image and blobs

    im_th = pre.rotate_bound(image=im_th, angle=skew_angle)
    blobs = loc.rotate_blobs(blobs=blobs,angle=skew_angle,old_dims=im.shape,new_dims=im_th.shape)
    #im_median = pre.rotate_bound(image=im_median, angle=skew_angle, border_white=True)

    ###
    
    #show(im_th)

    ### sort blobs according to columns pattern

    #blobs = blob_detector.blob_location(im_median)
    blob_list = loc.sort_points(blobs, column_number = black_dots_columns )

    ###

    #print(blob_list)



    ### iterate over all fivers(part of image with 5 answers), to create array of all answer lines

    fivers = fiver_iterator(im_th, blob_list)

    answer_ims = []
    for fiver in fivers:

        ### resize fivers to standart dimensions
        fiver = pre.resize(fiver, config.fiver_shape)

        ### remove margins from fiver image
        fiver = fiver[config.fiver_margin1:-config.fiver_margin1,
                      config.fiver_margin2:-config.fiver_margin2]

        #show(fiver)

        ### iterate over all answer_lines in a fiver

        fiver_fifth = int(fiver.shape[0] / 5)

        for answer_row_number in range(5):
            answer_row = fiver[answer_row_number * fiver_fifth: (answer_row_number+1) * fiver_fifth]
            answer_ims.append([answer_row])

            #show(answer_row)
            #plt.imsave("./answer_images/5/"+str(uuid4())+".png",answer_row, cmap="gray")
        
        ###

    answer_ims = np.array(answer_ims)   

    ###

    ### predict all answers using a neural net
    prediction = net.numpy_forward(answer_ims)


    #print(prediction)
    #for c,i in enumerate(np.argmax(prediction, axis = 1)):
    #    print(c + 1,"\t", i + 1)
    #print("sums" ,np.sum(prediction, axis = 0))

    return prediction
