import cv2 
import argparse
from configs import *
from FM2S import FM2S

def config_map(config):
    if config.lower() == 'confocal':
        return CONFOCAL_CONFIG
    elif config.lower() == 'twophoton':
        return TWOPHOTON_CONFIG
    elif config.lower() == 'widefield':
        return WIDEFIELD_CONFIG
    else:
        return BASE_CONFIG

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="BASE")
    parser.add_argument("-i", "--input_image_path", type=str, required=True)
    parser.add_argument("-o", "--output_image_path", type=str, default='output.png')
    args = parser.parse_args()
    return args

def main(args):
    raw = cv2.imread(args.input_image_path, cv2.IMREAD_GRAYSCALE)
    config = config_map(args.config)
    denoised = FM2S(raw, config)
    print("Finished!")
    cv2.imwrite(args.output_image_path, denoised)

if __name__ == "__main__":
    args = get_args()
    main(args)