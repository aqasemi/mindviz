
import os
from PIL import Image
from torchvision import transforms
import argparse

# def get_args_parser():
#     parser = argparse.ArgumentParser('train', add_help=False)
#     parser.add_argument('--type', type=str)
#     return parser.parse_args()

# args = get_args_parser()
class args:
    type = 'eeg' 

if args.type == 'eeg':
    data_dir = '/ibex/user/qasemiaa/datasets/things_eeg/image_set'
    save_dir = '/ibex/user/qasemiaa/datasets/things_eeg/image_set_resize'
elif args.type == 'meg':
    data_dir = 'data/things-meg/Image_set'
    save_dir = 'data/things-meg/Image_set_Resize'

os.makedirs(save_dir,exist_ok=True)
image_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(root, file))

t1 = transforms.Resize((224,224))

for path in image_paths:
    img = Image.open(path)

    img = t1(img)

    save_path = os.path.join(save_dir,path.split('/image_set/',1)[-1])
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    img.save(save_path)