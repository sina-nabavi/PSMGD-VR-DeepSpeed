import os
import re
import glob
import ast
import numpy as np

def performance(log_dir='./', output_dir='./averages/'):
    os.makedirs(output_dir, exist_ok=True)
    best_result_pattern = re.compile(r"Best Result: Epoch \d+, result ({.+})")

    # Gather all log files, ignoring the seed number.
    all_files = glob.glob(os.path.join(log_dir, 'PSMGDVR*.txt'))
    groups = {}
    for f in all_files:
        group_key = re.sub(r'_seed_\d+', '', os.path.basename(f))
        groups.setdefault(group_key, []).append(f)

    for group, files in groups.items():
        segmentation, depth, normal = [], [], []
        for file in files:
            with open(file, 'r') as fp:
                content = fp.read()
                result_match = best_result_pattern.search(content)
                if result_match:
                    result_dict = ast.literal_eval(result_match.group(1))
                    segmentation.append(result_dict['segmentation'])
                    depth.append(result_dict['depth'])
                    normal.append(result_dict['normal'])

        segmentation_mean = np.mean(segmentation, axis=0)
        depth_mean = np.mean(depth, axis=0)
        normal_mean = np.mean(normal, axis=0)

        with open(os.path.join(output_dir, group.replace('.txt', '') + '.txt'), 'w') as out_fp:
            out_fp.write(f"Average Best Results ({len(files)} seeds):\n")
            out_fp.write(f"segmentation: {segmentation_mean.tolist()}\n")
            out_fp.write(f"depth: {depth_mean.tolist()}\n")
            out_fp.write(f"normal: {normal_mean.tolist()}\n")


def convergence_test(log_dir='./logs', output_dir='./convergence_results',
                     seg_threshold=1.114, depth_threshold=0.448, normal_threshold=0.144): 
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Correct & safe regex, using .*? to capture each desired value clearly.
    epoch_pattern = re.compile(r"Epoch: \d+ \| TRAIN: .*? \| Time: ([\d\.]+) \| TEST: ([\d\.]+) .*?\| ([\d\.]+) .*?\| ([\d\.]+) .*? \| Time: ([\d\.]+)")

    all_files = glob.glob(os.path.join(log_dir, 'PSMGD*.txt'))

    groups = {}
    for f in all_files:
        group_key = re.sub(r'_seed_\d+', '', os.path.basename(f))
        groups.setdefault(group_key, []).append(f)

    for group, files in groups.items():
        mean_epoch_times, seg_times, depth_times, normal_times = [], [], [], []

        for file in files:
            seg_reached, depth_reached, normal_reached = None, None, None
            cumulative_time = 0
            epoch_times = []
            pre_reached_flags = [False, False, False]

            with open(file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    match = epoch_pattern.search(line)
                    if match:
                        
                        train_time, seg_loss, depth_loss, normal_loss, test_time = map(float, match.groups())
                        cumulative_time += (train_time + test_time)
                        epoch_times.append(train_time + test_time)

                        if not pre_reached_flags[0] and seg_loss <= seg_threshold:
                            seg_reached = cumulative_time / 60
                            pre_reached_flags[0] = True
                        if not pre_reached_flags[1] and depth_loss <= depth_threshold:
                            depth_reached = cumulative_time / 60
                            pre_reached_flags[1] = True
                        if not pre_reached_flags[2] and normal_loss <= normal_threshold:
                            normal_reached = cumulative_time / 60
                            pre_reached_flags[2] = True

                        if all(pre_reached_flags):
                            break

            if epoch_times: # prevent empty arrays
                mean_epoch_time = np.mean(epoch_times) / 60
                mean_epoch_times.append(mean_epoch_time)
                seg_times.append(seg_reached if seg_reached else np.nan)
                depth_times.append(depth_reached if depth_reached else np.nan)
                normal_times.append(normal_reached if normal_reached else np.nan)

        with open(os.path.join(output_dir, group.replace('.txt', '') + '_analysis.txt'), 'w') as fp:
            print(group)
            fp.write(f"Mean Epoch Time (train+test, minutes): {np.nanmean(mean_epoch_times):.2f}\n")
            fp.write(f"Time to reach TEST Segmentation Loss {seg_threshold}: {np.nanmean(seg_times):.2f} min\n")
            fp.write(f"Time to reach TEST Depth Loss {depth_threshold}: {np.nanmean(depth_times):.2f} min\n")
            fp.write(f"Time to reach TEST Normal Loss {normal_threshold}: {np.nanmean(normal_times):.2f} min\n")

#convergence_test(log_dir='./PSMGDVR/psmgd_VR_periodic/', output_dir='./PSMGDVR/psmgd_VR_periodic/convergence')

def sampel_complexity(log_dir='./logs', output_dir='./convergence_results',
                      seg_threshold=1.114, depth_threshold=0.448, normal_threshold=0.144): 
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Updated regex to capture the epoch number as well.
    epoch_pattern = re.compile(r"Epoch: (\d+) \| TRAIN: .*? \| Time: ([\d\.]+) \| TEST: ([\d\.]+) .*?\| ([\d\.]+) .*?\| ([\d\.]+) .*? \| Time: ([\d\.]+)")

    all_files = glob.glob(os.path.join(log_dir, 'PSMGD*.txt'))

    groups = {}
    for f in all_files:
        group_key = re.sub(r'_seed_\d+', '', os.path.basename(f))
        groups.setdefault(group_key, []).append(f)

    for group, files in groups.items():
        mean_epoch_times, seg_epochs, depth_epochs, normal_epochs = [], [], [], []

        for file in files:
            seg_reached_epoch, depth_reached_epoch, normal_reached_epoch = None, None, None

            with open(file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    match = epoch_pattern.search(line)
                    if match:
                        
                        epoch_num, train_time, seg_loss, depth_loss, normal_loss, test_time = map(float, match.groups())
                        if seg_reached_epoch is None and seg_loss <= seg_threshold:
                            seg_reached_epoch = epoch_num
                        if depth_reached_epoch is None and depth_loss <= depth_threshold:
                            depth_reached_epoch = epoch_num
                        if normal_reached_epoch is None and normal_loss <= normal_threshold:
                            normal_reached_epoch = epoch_num

                        if seg_reached_epoch and depth_reached_epoch and normal_reached_epoch:
                            break

            seg_epochs.append(seg_reached_epoch + 1 if seg_reached_epoch is not None else np.nan)
            depth_epochs.append(depth_reached_epoch + 1 if depth_reached_epoch is not None else np.nan)
            normal_epochs.append(normal_reached_epoch + 1 if normal_reached_epoch is not None else np.nan)
        print(file)
        with open(os.path.join(output_dir, group.replace('.txt', '') + '_analysis.txt'), 'w') as fp:
            fp.write(f"Mean Epoch Number to reach TEST Segmentation Loss {seg_threshold}: {np.nanmean(seg_epochs) * 794:.2f}\n")
            fp.write(f"Mean Epoch Number to reach TEST Depth Loss {depth_threshold}: {np.nanmean(depth_epochs) * 794:.2f}\n")
            fp.write(f"Mean Epoch Number to reach TEST Normal Loss {normal_threshold}: {np.nanmean(normal_epochs) * 794:.2f}\n")


def compute_dM_and_MR(log_dir='./logs', output_dir='./metrics_results',
                      baseline_results=[0.54,0.75,0.38,0.15,23.4,16.9,0.35,0.61,0.72]):
    os.makedirs(output_dir, exist_ok=True)
    error_pattern = re.compile(r'Best Result: Epoch \d+, result ({.*?})')
    directions = np.array([1,1,0,0,0,0,1,1,1]) # 1: higher better, 0: lower better 

    groups = {}
    for f in glob.glob(os.path.join(log_dir, 'output_weighting_*.txt')):
        group_key = re.sub(r'_seed_\d+', '', os.path.basename(f))
        groups.setdefault(group_key, []).append(f)

    group_errors = {}
    for group, files in groups.items():
        errors = []
        for file in files:
            with open(file, 'r') as f_in:
                content = f_in.read()
                match = error_pattern.search(content)
                if match:
                    result = eval(match.group(1))
                    errs = result['segmentation'] + result['depth'] + result['normal']
                    errors.append(errs)
        avg_errs = np.mean(errors, axis=0)
        group_errors[group] = avg_errs

    method_names = list(group_errors.keys())
    method_errors = np.array(list(group_errors.values()))
    ranks = np.zeros_like(method_errors, dtype=float)

    # Corrected ranking logic here
    for i, higher_better in enumerate(directions):
        col = method_errors[:, i]
        if higher_better:
            # Rank best to worst (higher is better)
            temp_rank = col.argsort()[::-1].argsort() + 1
        else:
            # Rank best to worst (lower is better)
            temp_rank = col.argsort().argsort() + 1
        ranks[:, i] = temp_rank

    MR = ranks.mean(axis=1)

    MB = np.array(baseline_results)
    dM = (((-1)**directions)*((method_errors-MB)/MB)).mean(axis=1)*100

    for idx, method in enumerate(method_names):
        with open(os.path.join(output_dir, method+'_metrics.txt'), 'w') as f_out:
            f_out.write(f"MR: {MR[idx]:.3f}\n")
            f_out.write(f"dM%: {dM[idx]:.3f}%\n")
# Example usage:
sampel_complexity(log_dir='./PSMGDVR_lower_lr_e-5/samplec/', output_dir='./PSMGDVR_lower_lr_e-5/samplecout')
# usage example
#performance(log_dir='./PSMGDVR/psmgd_VR_periodic/', output_dir='./PSMGDVR/psmgd_VR_periodic/average')
import ast
def parse_log_write_output(input_dir='./logs', output_dir='./parsed_outputs'):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"segmentation: (\[.*?\])\s+depth: (\[.*?\])\s+normal: (\[.*?\])", re.DOTALL)
    
    for log_file in glob.glob(os.path.join(input_dir, '*.txt')):
        with open(log_file, 'r') as f:
            log = f.read()
            match = pattern.search(log)
            if match:
                seg = ast.literal_eval(match.group(1))
                dep = ast.literal_eval(match.group(2))
                norm = ast.literal_eval(match.group(3))

                output_line = "& {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & - & - \\\\".format(
                    seg[0]*100, seg[1]*100, dep[0], dep[1], norm[0], norm[1], norm[2]*100, norm[3]*100, norm[4]*100
                )

                output_filename = os.path.join(
                    output_dir, os.path.basename(log_file).replace('.txt', '_parsed.txt')
                )
                with open(output_filename, 'w') as out_fp:
                    out_fp.write(output_line + '\n')
# parse_log_write_output(input_dir='./PSMGDVR/baselines/averaged_results/Automatic/', output_dir='./PSMGDVR/baselines/averaged_results/Automatic/out/')
# compute_dM_and_MR(log_dir='./PSMGDVR/baselines/', output_dir='./PSMGDVR/baselines/deltaM',
#                  baseline_results=[0.5304,0.7525,0.4038,0.1655,25.02,19.11,0.3065,0.5666,0.6862])


def save_model_counter(log_dir):
    for log_file in glob.glob(f"{log_dir}/*.txt"):
        with open(log_file, 'r+') as f:
            content = f.read()
            n_saves = len(re.findall(r'^Save Model.*?$', content, re.MULTILINE))
            f.write(f"\nNumber of saves: {n_saves}\n")

def average_num_saves(log_dir='./logs'):
    log_files = glob.glob(f"{log_dir}/*.txt")
    saves = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            last_line = f.readlines()[-1]
            match = re.search(r'Number of saves: (\d+)', last_line)
            if match:
                saves.append(int(match.group(1)))
    if saves:
        print(f'Average number of saves: {np.mean(saves):.2f}')
        print(f'Minimum number of saves: {np.min(saves)}')
        print(f'Maximum number of saves: {np.max(saves)}')
    else:
        print('No save numbers found.')

# average_num_saves('./PSMGDVR/baselines/')

#save_model_counter(log_dir='./PSMGDVR/baselines/')

import re
import matplotlib.pyplot as plt

def extract_test_segmentation_loss(log_file_path):
    epoch_nums = []
    test_losses = []

    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith("Epoch:"):
                # Extract epoch number
                match_epoch = re.search(r"Epoch:\s+(\d+)", line)
                # Extract test segmentation loss (after '| TEST:' and before the next '|')
                match_test_loss = re.search(r"\|\s*TEST:\s*([\d.]+)", line)

                if match_epoch and match_test_loss:
                    epoch = int(match_epoch.group(1))
                    loss = float(match_test_loss.group(1))
                    epoch_nums.append(epoch)
                    test_losses.append(loss)

    return epoch_nums, test_losses

def plot_test_loss(epochs, losses, output_pdf):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label='Test Segmentation Loss')
    plt.title('Segmentation Test Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Segmentation Test Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_pdf)
    print(f"Plot saved to {output_pdf}")

# Usage
# log_file = './PSMGDVR/baselines/output_weighting_PSMGDVR_n_8_weighting_EW_seed_811637239.txt'  # Change to your actual log file path
# output_pdf = './plot/test_segmentation_loss_ew_highlr.pdf'
# epochs, losses = extract_test_segmentation_loss(log_file)
# plot_test_loss(epochs, losses, output_pdf)