
import Augmentor
import os
from PIL import Image

# set path and numbers of the augmented images 
input_folder = "/Users/zhangqian/dental_classifier_hti/normal/no inflamation"
output_folder = "/Users/zhangqian/dental_classifier_hti/dentalclassifier/Dentaldata/healthyteeth"
num_generated_images = 1000


# create a pipe 
p = Augmentor.Pipeline(input_folder, output_folder)

# image augment
p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_left_right(probability=0.5)

# set the format
p.set_save_format("PNG")

# excute and save
p.sample(num_generated_images, multi_threaded=False) 

# rename the images from 1-1000, and make it 300x300 size
output_images = os.listdir(output_folder)
output_images.sort()  
for i, image_name in enumerate(output_images):
    
    img = Image.open(os.path.join(output_folder, image_name))
    
    img = img.convert("RGB")
    
    img = img.resize((300, 300), Image.LANCZOS)
    
    new_image_name = f"{i + 1}.png"
    img.save(os.path.join(output_folder, new_image_name), "PNG")
    
    img.close()
    
    os.remove(os.path.join(output_folder, image_name))

# transform .png to .jpg
healthy_teeth_folder = "/Users/zhangqian/dental_classifier_hti/dentalclassifier/Dentaldata/healthyteeth"

for filename in os.listdir(healthy_teeth_folder):
    if filename.endswith(".png"): 
        png_path = os.path.join(healthy_teeth_folder, filename)
        img = Image.open(png_path)
        jpg_path = os.path.join(healthy_teeth_folder, filename.replace(".png", ".jpg"))
        img.save(jpg_path, "JPEG")
        os.remove(png_path)

