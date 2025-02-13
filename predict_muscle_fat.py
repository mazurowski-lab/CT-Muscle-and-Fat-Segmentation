import torch
import os
import warnings
import argparse
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def segment_vertebrae(input_path, output_path):
    """
    Perform vertebrae segmentation on the given input NIfTI file and save the results.
    """
    # Create output directory path
    output_path = os.path.join(output_path, f"{os.path.basename(input_path).replace('.nii.gz', '')}_vertebrae_segmentation")
    os.makedirs(output_path, exist_ok=True)
    
    # Use specific options to avoid environment issues
    command = (f"TotalSegmentator -i {input_path} -o {output_path} "
              f"--roi_subset vertebrae_T12 vertebrae_L3 vertebrae_L4")
    print(f"Running command: {command}")
    os.system(command)
        
def calculate_2D_metrics(input_nii, segmentation_nii, vertebrae_path):
    """
    Calculate 2D body composition metrics at L3 level
    """
    # Load images
    img = nib.load(input_nii)
    seg = nib.load(segmentation_nii)
    
    # Get spacing information
    spacing = img.header.get_zooms()
    
    # Load L3 vertebra mask
    l3_mask = nib.load(os.path.join(vertebrae_path, "vertebrae_L3.nii.gz"))
    
    # Find the L3 slice
    l3_data = l3_mask.get_fdata()
    l3_slice = np.argmax(np.sum(l3_data, axis=(0,1)))
    
    # Get the image data at L3 level
    img_data = img.get_fdata()[:,:,l3_slice]
    seg_data = seg.get_fdata()[:,:,l3_slice]
    
    # Calculate metrics
    metrics = {
        'filename': os.path.basename(input_nii),
        'muscle_area_mm2': np.sum(seg_data == 1) * spacing[0] * spacing[1],
        'muscle_density_hu': np.mean(img_data[seg_data == 1]),
        'sfat_area_mm2': np.sum(seg_data == 2) * spacing[0] * spacing[1],
        'vfat_area_mm2': np.sum(seg_data == 3) * spacing[0] * spacing[1],
        'mfat_area_mm2': np.sum(seg_data == 4) * spacing[0] * spacing[1],
        'total_fat_area_mm2': np.sum(np.isin(seg_data, [2,3,4])) * spacing[0] * spacing[1],
        'body_area_mm2': np.sum(img_data > -500) * spacing[0] * spacing[1]
    }
    
    return metrics

def calculate_3D_metrics(input_nii, segmentation_nii, vertebrae_path):
    """
    Calculate 3D body composition metrics between T12 and L4
    """
    # Load images
    img = nib.load(input_nii)
    seg = nib.load(segmentation_nii)
    spacing = img.header.get_zooms()
    
    # Load vertebrae masks
    t12_mask = nib.load(os.path.join(vertebrae_path, "vertebrae_T12.nii.gz"))
    l4_mask = nib.load(os.path.join(vertebrae_path, "vertebrae_L4.nii.gz"))
    
    # Find the range between T12 and L4
    t12_data = t12_mask.get_fdata()
    l4_data = l4_mask.get_fdata()
    
    t12_slice = np.argmax(np.sum(t12_data, axis=(0,1)))
    l4_slice = np.argmax(np.sum(l4_data, axis=(0,1)))
    
    slice_range = range(min(t12_slice, l4_slice), max(t12_slice, l4_slice) + 1)
    
    # Get image data
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    # Calculate volumes and densities
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    metrics = {
        'filename': os.path.basename(input_nii),
        'muscle_volume_mm3': np.sum(seg_data[:,:,slice_range] == 1) * voxel_volume,
        'muscle_density_hu': np.mean(img_data[:,:,slice_range][seg_data[:,:,slice_range] == 1]),
        'sfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 2) * voxel_volume,
        'vfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 3) * voxel_volume,
        'mfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 4) * voxel_volume,
        'total_fat_volume_mm3': np.sum(np.isin(seg_data[:,:,slice_range], [2,3,4])) * voxel_volume,
        'body_volume_mm3': np.sum(img_data[:,:,slice_range] > -500) * voxel_volume,
        'body_height_mm': len(slice_range) * spacing[2]
    }
    
    return metrics

def save_metrics_to_csv(metrics_list, output_path, type_2d=True):
    """
    Save metrics to CSV file
    """
    df = pd.DataFrame(metrics_list)
    filename = 'body_composition_2d.csv' if type_2d else 'body_composition_3d.csv'
    csv_path = os.path.join(output_path, filename)
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

def main(args):
    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Define nnUNet model path correctly
    model_path = 'nnUNetTrainer__nnUNetResEncUNetXLPlans__2d'

    # Choose checkpoint file based on user input
    checkpoint_name = 'checkpoint_best.pth' if "best" in args.checkpoint_type else 'checkpoint_final.pth'
    
    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=(5,),  # Ensure you are using the correct fold(s)
        checkpoint_name=checkpoint_name,
    )
    
    # Define input and output paths from arguments
    input_path = args.input
    output_path = args.output

    # Check input type
    if os.path.isfile(input_path):
        input_files = [[input_path]]  # Single file in nested list
        if os.path.isdir(output_path):
            output_files = [join(output_path, os.path.basename(f).replace(".nii.gz", "_segmentation.nii.gz")) for f in input_files[0]]
        else:
            output_files = [output_path]  # Single output file
    elif os.path.isdir(input_path):
        input_files = [[f] for f in subfiles(input_path, suffix='.nii.gz', join=True)]
        if not input_files:
            raise FileNotFoundError(f"No NIfTI files found in {input_path}")
        if os.path.isfile(output_path):
            raise ValueError("Output cannot be a file when input is a folder.")
        os.makedirs(output_path, exist_ok=True)
        output_files = [join(output_path, os.path.basename(f).replace(".nii.gz", "_segmentation.nii.gz")) for f in input_files[0]]
    else:
        raise ValueError("Input path must be either a valid file or directory.")
    
    # Perform prediction
    predictor.predict_from_files(
        input_files,
        output_files,
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        num_processes_preprocessing=len(input_files),  # Reduce processes to avoid RAM overuse
        num_processes_segmentation_export=len(output_files),
        folder_with_segs_from_prev_stage=None,
        num_parts=4,
        part_id=0
    )

    print("Segmentation completed! Results saved in:", output_path)

    # Calculate body composition metrics only if requested
    if args.body_composition_type != "None":
        output_body_composition = args.output_body_composition if args.output_body_composition else (
            output_path if os.path.isdir(output_path) else "results"
        )
        os.makedirs(output_body_composition, exist_ok=True)
        
        for input_file in input_files[0]:
                segment_vertebrae(input_file, output_body_composition)

    if args.body_composition_type == "2D" or args.body_composition_type == "both":
        # Calculate 2D body composition metrics
        metrics_2d = []
        for input_file in input_files[0]:
            # Get correct paths for segmentation and vertebrae files
            base_name = os.path.basename(input_file).replace('.nii.gz', '')
            seg_file = os.path.join(output_path, f"{base_name}_segmentation.nii.gz")
            vertebrae_path = os.path.join(output_body_composition, 
                                        f"{base_name}_vertebrae_segmentation")
            
            try:
                if os.path.exists(seg_file) and os.path.exists(vertebrae_path):
                    metrics = calculate_2D_metrics(input_file, seg_file, vertebrae_path)
                    metrics_2d.append(metrics)
                else:
                    print(f"Missing files for {input_file}:")
                    print(f"Segmentation file exists: {os.path.exists(seg_file)}")
                    print(f"Vertebrae path exists: {os.path.exists(vertebrae_path)}")
            except Exception as e:
                print(f"Error processing {input_file} for 2D metrics: {str(e)}")
        
        if metrics_2d:  # Only save if we have metrics
            save_metrics_to_csv(metrics_2d, output_body_composition, type_2d=True)
        else:
            print("No 2D metrics were calculated successfully")
        
    if args.body_composition_type == "3D" or args.body_composition_type == "both":
        # Calculate 3D body composition metrics
        metrics_3d = []
        for input_file in input_files[0]:
            # Get correct paths for segmentation and vertebrae files
            base_name = os.path.basename(input_file).replace('.nii.gz', '')
            seg_file = os.path.join(output_path, f"{base_name}_segmentation.nii.gz")
            vertebrae_path = os.path.join(output_body_composition,
                                        f"{base_name}_vertebrae_segmentation")
            
            try:
                if os.path.exists(seg_file) and os.path.exists(vertebrae_path):
                    metrics = calculate_3D_metrics(input_file, seg_file, vertebrae_path)
                    metrics_3d.append(metrics)
                else:
                    print(f"Missing files for {input_file}:")
                    print(f"Segmentation file exists: {os.path.exists(seg_file)}")
                    print(f"Vertebrae path exists: {os.path.exists(vertebrae_path)}")
            except Exception as e:
                print(f"Error processing {input_file} for 3D metrics: {str(e)}")
        
        if metrics_3d:  # Only save if we have metrics
            save_metrics_to_csv(metrics_3d, output_body_composition, type_2d=False)
        else:
            print("No 3D metrics were calculated successfully")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run nnUNet segmentation on muscle and fat MRI/CT images.")

    # Define command-line arguments
    parser.add_argument("--input", type=str, default="demo", help="Path to input file or folder containing .nii.gz files (default: demo)")
    parser.add_argument("--output", type=str, default="results", help="Path to output file or folder for segmentation results (default: results)")
    parser.add_argument("--save_probabilities", type=bool, default=False, help="Whether to save probability maps (default: False)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite existing predictions (default: False)")
    parser.add_argument("--checkpoint_type", type=str, default="final", help="Specify 'best' to use checkpoint_best.pth or 'final' to use checkpoint_final.pth (default: final)")
    parser.add_argument("--body_composition_type", type=str, default="both", help="Specify '2D' for 2D body composition, '3D' for 3D body composition, 'both' for both, 'None' for no body composition (default: both)")
    parser.add_argument("--output_body_composition", type=str, help="Path to output body composition metric file. If --output is a folder, this will be set to the same folder. Otherwise, it defaults to 'results'.""Path to output body composition metric file")
    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args)