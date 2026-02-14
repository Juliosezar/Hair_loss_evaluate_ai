import os
import sys
from rembg import remove

input_dir = '../dataset/1- raw/'
output_dir = '../dataset/2- remove_bg/'


def remove_bg(input_dir, output_dir):
    for i in os.listdir(input_dir):
        try:
            print(i)
            input_path = os.path.join(input_dir, i)
            output_path = os.path.join(output_dir, i)
            

            with open(input_path, 'rb') as i:
                with open(output_path, 'wb') as o:
                    input = i.read()
                    output = remove(input)
                    o.write(output)
        except:
            print('error')
            continue


if __name__ == "__main__":
    remove_bg(input_dir, output_dir)
