import os
import subprocess

dataset = os.listdir('Dataset/')
# dataset = ['Test', 'Train.txt', 'Validation.txt', 'Train', 'Test.txt', 'Validation']

def split_video(video_file, image_name_prefix, source_path, destination_path):
    input_file_path = os.path.join(source_path, video_file)
    output_file_path = os.path.join(destination_path, image_name_prefix + '%d.jpg')
    cmd = f'ffmpeg -i "{input_file_path}" -vf fps=1 {output_file_path} -hide_banner'
    return subprocess.check_output(cmd , shell=True)

# def split_video(video_file, image_name_prefix, destination_path):
#     return subprocess.check_output('ffmpeg -i "' + destination_path+video_file + '" ' + image_name_prefix + '%d.jpg -hide_banner', shell=True, cwd=destination_path)

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
                        destination_path = os.path.abspath('../DAiSEE/Dataset/' + ttv + '/Images')
                        split_video(clip, clip[:-4], source_path, destination_path)
print ("================================================================================\n")
print ("Frame Extraction Successful")