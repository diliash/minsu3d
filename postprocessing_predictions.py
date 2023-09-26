import argparse
import glob
import os
import numpy as np
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict_dir', type=str, default='./output/PartNetSim/PointGroup/zero_threshold_eo/inference/val/predictions/instance', help='the directory of the predictions')
    
    args = parser.parse_args()

    model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{args.predict_dir}/*.txt")]
    print(model_ids)

    for MODELID in tqdm.tqdm(model_ids):
        with open(f"{args.predict_dir}/{MODELID}.txt", "r") as f:
            lines = f.readlines()
        
        all_base = np.empty([])
        all_masked = np.empty([])
        first_base = None
        len_file = 0

        for line in lines:
            mask_file, prediction, _ = line.split(" ")
            with open(f"{args.predict_dir}/{mask_file}", "r") as f:
                point_masked = f.readlines()
            
            len_file = len(point_masked)

            if prediction == "3":
                if not first_base:
                    first_base = mask_file
                for index, point in point_masked:
                    if "1" in point:
                        all_base = np.append(all_base, index)
                        all_masked = np.append(all_masked, index)
            else:
                for index, point in point_masked:
                    if "1" in point:
                        all_masked = np.append(all_masked, index)
            

        with open(f"{args.predict_dir}/{first_base}", "w") as f:
            for i in range(len_file):
                if i in all_base or i not in all_masked:
                    f.write("1\n")
                else:
                    f.write("0\n")

        
        with open(f"{args.predict_dir}/{MODELID}.txt", "w") as f:
            for line in lines:
                mask_file, prediction, _ = line.split(" ")
                if prediction != "3" or mask_file == first_base:
                    f.write(line)


        
