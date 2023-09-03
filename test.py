import os

def rename_files_in_folder(folder_path):
    """
    Rename files in the specified folder by removing "frame_" from their names.
    
    Parameters:
    - folder_path (str): Path to the folder containing the files to rename.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    
    for file_name in files:
        # Check if "frame_" exists in the file name
        if "frame_" in file_name:
            # Create the new name by replacing "frame_" with an empty string
            new_file_name = file_name.replace("frame_", "")
            # Get full paths for the old and new file names
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f'Renamed {file_name} to {new_file_name}')

if __name__ == "__main__":
    # Example usage:
    folder_path = "vmd_code\\code\\VMD\\more2\\646"
    rename_files_in_folder(folder_path)

    folder_path = "vmd_code\\code\\VMD\\more2\\405"
    rename_files_in_folder(folder_path)
