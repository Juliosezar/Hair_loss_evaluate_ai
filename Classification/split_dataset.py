
import splitfolders

def split_dataset(input_dir, output_dir):
    splitfolders.ratio(input_dir, output=output_dir, seed=42, ratio=(0.8, 0.2))
    print("dataset splited successfully!")

if __name__ == "__main__":
    input_dir = './dataset/4- labled/'
    output_dir = './dataset/5- slplited/'
    split_dataset(input_dir, output_dir)


