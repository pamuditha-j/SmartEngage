import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.image import imread

dataset = os.listdir('Dataset/')

def split_video(video_file, image_name_prefix, source_path, destination_path):
    input_file_path = os.path.join(source_path, video_file)
    output_file_path = os.path.join(destination_path, image_name_prefix + '%d.jpg')
    cmd = f'ffmpeg -i "{input_file_path}" -vf fps=1 {output_file_path} -hide_banner'
    subprocess.check_output(cmd, shell=True)

def display_frames(destination_path, num_frames=5):
    image_files = [f for f in os.listdir(destination_path) if f.endswith('.jpg')][:num_frames]

    for img_file in image_files:
        img_path = os.path.join(destination_path, img_file)
        img = imread(img_path)
        plt.imshow(img)
        plt.title(img_file)
        plt.show()

for ttv in dataset:
    if(ttv != '.DS_Store'):
        users = os.listdir('Dataset/'+ttv+'/')

        for user in users:
            if(user != '.DS_Store' and user != 'Images'):
              currUser = os.listdir('Dataset/'+ttv+'/'+user+'/')
              for extract in currUser:
                    if(extract != '.DS_Store'):
                        clip = os.listdir('Dataset/'+ttv+'/'+user+'/'+extract+'/')[0]
                        print (clip[:-4])
                        source_path = os.path.abspath('.')+'/Dataset/'+ttv+'/'+user+'/'+extract+'/'
                        destination_path = os.path.abspath('../Dataset/Image_Dataset_2/' + ttv + '/')

                        if not os.path.exists(destination_path):
                            os.makedirs(destination_path)

                        split_video(clip, clip[:-4], source_path, destination_path)
                        display_frames(destination_path, num_frames=5)

print("================================================================================\n")
print("Frame Extraction and Display Successful")
