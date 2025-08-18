import os
import shutil
import sys
import time # Needed for modification time comparison

# --- Configuration ---
# !! Adjust these paths if your directories are named differently !!
SOURCE_BASE_DIR = 'harness_results'
TARGET_BASE_DIR = 'organized_results'
TASK_NAMES = ['gsm8k', 'leaderboard', 'mmlu']
# --- End Configuration ---

def find_newest_result_file(search_paths):
    """
    Searches given directory paths for 'results_*.json' files and returns the
    full path to the newest one based on modification time.

    Args:
        search_paths (list): A list of directory paths to search within.

    Returns:
        str or None: The full path to the newest result file, or None if no
                     result files are found.
    """
    newest_file = None
    latest_mtime = 0
    
    

    for search_path in search_paths:
        if not os.path.isdir(search_path):
            # print(f"  Debug: Path {search_path} is not a directory, skipping.")
            continue
        try:
            # print(f"  Debug: Searching in directory: {search_path}")
            for filename in os.listdir(search_path):
                if filename.startswith('results_') and filename.endswith('.json'):
                    full_path = os.path.join(search_path, filename)
                    if os.path.isfile(full_path): # Ensure it's a file
                        try:
                            mtime = os.path.getmtime(full_path)
                            # print(f"  Debug: Found result file: {full_path} (mtime: {mtime})")
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                                newest_file = full_path
                                # print(f"  Debug: New newest file found: {newest_file}")
                        except OSError as e:
                            print(f"  Warning: Could not get mtime for {full_path}: {e}")
                            continue # Skip if cannot get modification time
        except OSError as e:
             print(f"  Warning: Could not list directory {search_path}: {e}")
             continue # Skip if cannot list directory

    # print(f"  Debug: Final newest file for search paths: {newest_file}")
    return newest_file


def organize_results(source_dir, target_dir, tasks):
    """
    Organizes benchmark results from a source directory into a structured
    target directory, handling results possibly nested one level deeper
    within task directories.

    Args:
        source_dir (str): The path to the root directory containing the results.
        target_dir (str): The path to the directory for organized results.
        tasks (list): A list of task names (directory names) to look for.
    """
    print(f"Starting organization...")
    print(f"Source directory: {os.path.abspath(source_dir)}")
    print(f"Target directory: {os.path.abspath(target_dir)}")
    print(f"Tasks to process: {', '.join(tasks)}")
    print("-" * 30)

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        sys.exit(1) # Exit the script if source doesn't exist

    os.makedirs(target_dir, exist_ok=True)
    copied_count = 0
    skipped_count = 0
    # Keep track of task paths already fully processed to avoid duplication
    # This is important because os.walk will visit parent dirs then child dirs
    processed_task_paths = set()

    # Walk through the source directory
    # topdown=True allows us to modify `dirs` to prune the walk
    for root, dirs, files in os.walk(source_dir, topdown=True):
        current_dir_name = os.path.basename(root)

        # Check if this directory is one of the tasks we care about
        if current_dir_name in tasks:
            task_root_path = root

            # Avoid reprocessing if we already handled this path via a parent walk iteration
            # which found results within its subdirs.
            if task_root_path in processed_task_paths:
                # We've already dealt with finding the results within this structure when we
                # processed its parent directory. Tell walk not to go into its dirs either.
                dirs[:] = []
                continue

            print(f"Found potential task directory: {task_root_path}")

            # --- Extract Model Name ---
            # Assumes structure like .../model_name/task_name/...
            path_parts = os.path.normpath(task_root_path).split(os.sep)
            model_name = None
            try:
                 # Find the index of the task name in the path parts
                 # Use reversed list search to find the last occurrence in case task names repeat
                 task_index = len(path_parts) - 1 - path_parts[::-1].index(current_dir_name)
                 if task_index > 0:
                     # Model name is the part right before the task name
                     model_name = path_parts[task_index - 1]
                 else:
                     print(f"  Warning: Task '{current_dir_name}' found at unexpected level: {task_root_path}. Skipping.")
                     skipped_count += 1
                     dirs[:] = [] # Don't descend further from here
                     continue

            except (ValueError, IndexError):
                 print(f"  Warning: Could not reliably determine model name structure for task path '{task_root_path}'. Skipping.")
                 skipped_count += 1
                 dirs[:] = [] # Don't descend further from here
                 continue
            # --- End Model Name Extraction ---


            # --- Search for the newest results_*.json ---
            # Define paths to search:
            # 1. The task directory itself (root)
            # 2. All immediate subdirectories within the task directory
            paths_to_search = [task_root_path] + [os.path.join(task_root_path, d) for d in dirs]
            print(f"  Searching for results in: '{task_root_path}' and its immediate subdirectories: {dirs}")

            newest_source_file_path = find_newest_result_file(paths_to_search)
            # --- End Search ---

            if newest_source_file_path:
                print(f"  Found newest result file: {newest_source_file_path}")

                # --- Add path and subdirs to processed set ---
                # Mark this task path as processed so we don't re-evaluate if walk visits it again.
                # Also mark its immediate children, as we explicitly searched them.
                processed_task_paths.add(task_root_path)
                for d in dirs:
                    processed_task_paths.add(os.path.join(task_root_path, d))
                # --- End processed set update ---

                # --- Construct target path and Copy ---
                task_name = current_dir_name # Use the identified task name
                target_model_dir = os.path.join(target_dir, task_name, model_name)
                target_file_path = os.path.join(target_model_dir, 'result.json')

                # Create target directory structure
                print(f"  Creating target directory: {target_model_dir}")
                os.makedirs(target_model_dir, exist_ok=True)

                # Copy the file
                try:
                    print(f"  Copying '{newest_source_file_path}' to '{target_file_path}'")
                    shutil.copy2(newest_source_file_path, target_file_path)
                    copied_count += 1
                except Exception as e:
                    print(f"  Error copying file: {e}")
                    skipped_count += 1
                # --- End Copy ---

                # --- Prune Walk ---
                # IMPORTANT: Since we've found the definitive newest result for this task structure
                # (by searching the task dir and its immediate subdirs), tell os.walk
                # NOT to descend into these subdirectories (`dirs`) any further.
                # This prevents redundant checks or potential errors if subdirs also match task names.
                print(f"  Pruning walk traversal into subdirectories: {dirs}")
                dirs[:] = []
                # --- End Prune ---

            else:
                # No result files found in the task directory or its immediate children
                print(f"  No 'results_*.json' files found within '{task_root_path}' or its immediate subdirs. Skipping this task instance.")
                skipped_count += 1
                # We don't prune dirs here because maybe a valid task structure exists deeper down,
                # although the logic primarily relies on finding the task dir first.


    print("-" * 30)
    print(f"Organization complete.")
    print(f"Files copied: {copied_count}")
    print(f"Task instances skipped (no results found or error): {skipped_count}")

# --- Run the organization ---
if __name__ == "__main__":
    organize_results(SOURCE_BASE_DIR, TARGET_BASE_DIR, TASK_NAMES)