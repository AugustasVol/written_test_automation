#!/usr/bin/env python3
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Create a black image

### config
y = 30
x = 330

circle_line_thick = 2
circle_radius = 10
space_between_boxes_x = 50
space_between_boxes_y = 0

square_size = y

font_size = 20

font_file = os.path.dirname(os.path.abspath(__file__)) + "/font.ttf"
font = ImageFont.truetype(font_file, font_size)

question_letter_margin = 5

letter_array = ["A", "B", "C" ,"D", "E","F", "G","H", "I", "Y"]

### end config

def draw_circle(draw_obj,x,y,radius, line_width):
        draw_obj.ellipse([x-radius, y-radius, x+radius, y+radius], fill=0)
        draw_obj.ellipse([x-radius + line_width, y - radius + line_width, x + radius - line_width, y + radius - line_width], fill=(255,255,255))

def draw_square(draw_obj, x,y, extent):
    half = int(extent / 2)
    x1 = x- half
    y1 = y-half
    x2 = x + half
    y2 = y + half
    
    draw_obj.rectangle(  [(x1, y1), (x2, y2)], fill=0  )
    
def create_test_line(num, sections):

    sections = sections + 1 + 1


    section_size = int(x / sections + 1)
    middle_y = int(y / 2)

    img = np.ones((y,x, 3),dtype= np.uint8) * 255

    img = Image.fromarray(img, 'RGB')


    draw = ImageDraw.Draw(img)

    #draw.line((0, 0, x-1, 0), fill=0, width = outer_line_thick)
    #draw.line((0, 0, 0, y-1), fill=0, width = outer_line_thick)
    #draw.line((x-1,y-1, 0,y-1), fill=0, width = outer_line_thick)
    #draw.line((x-1,y-1, x-1,0), fill=0, width = outer_line_thick)




    for section_i in range(1,sections- 1):
        section_start_x = section_i * section_size
        section_middle_x = section_start_x + int(section_size / 2)

        draw_circle(draw, section_middle_x, middle_y, circle_radius, circle_line_thick)


    draw.text(( question_letter_margin ,middle_y - int(font_size/2)),str(num),font=font, fill=0)
    del draw


    return img

def create_top_letters(sections):

    sections = sections + 1 + 1

    img = np.ones((y,x, 3),dtype= np.uint8) * 255

    img = Image.fromarray(img, 'RGB')

    section_size = int(x / sections)
    middle_y = int(y / 2)

    draw = ImageDraw.Draw(img)

    section_start_x = 0 
    section_middle_x = section_start_x + int(section_size / 2)
    draw_square(draw, section_middle_x,middle_y, square_size)

    for section_i in range(1,sections -1 ):
        section_start_x = section_i * section_size
        section_middle_x = section_start_x + int(section_size / 2)
        draw.text(( section_middle_x,middle_y - int(font_size/2)), letter_array[section_i - 1],font=font, fill=0)

    section_start_x = (sections - 1) * section_size
    section_middle_x = section_start_x + int(section_size / 2)
    draw_square(draw, section_middle_x,middle_y, square_size)
    del draw

    return img

def create_5_sheet( sections, start_quesion_num = 1):
    margin = np.ones((space_between_boxes_y, x, 3), dtype=np.uint8) * 255

    top = np.array(create_top_letters(sections))

    img = top

    for q in range(start_quesion_num, start_quesion_num + 5):
        box = np.array(create_test_line(q, sections))
        img = np.concatenate((img, box), axis = 0)

    

    img = Image.fromarray(img, 'RGB')

    return img


def question_sheet(questions, show=True, save=True, output_name = "output.png"):


    if questions not in [i for i in range(10,51,10)]:
        raise


    half_questions = int(questions / 2)

    top = create_top_letters(5)

    first_half_fivers = []
    for i in range(1,half_questions + 1, 5):
        first_half_fivers.append(create_5_sheet(5,i))
    first_half_fivers.append(top)
    first_half = np.concatenate(first_half_fivers,axis=0)

    second_half_fivers = []
    for i in range(half_questions + 1,questions + 1, 5):
        second_half_fivers.append(create_5_sheet(5,i))
    second_half_fivers.append(top)
    second_half = np.concatenate(second_half_fivers, axis=0)

    middle_margin = np.ones((first_half.shape[0], space_between_boxes_x, 3), dtype=np.uint8) * 255


    img = np.concatenate((first_half,middle_margin, second_half), axis=1)

    img = Image.fromarray(img, "RGB")
    if save:
        img.save(output_name)
    if show:
        img.show()

    return img

if __name__ == "__main__":
    question_sheet(30)
