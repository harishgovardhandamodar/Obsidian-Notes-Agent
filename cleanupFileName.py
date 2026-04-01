import os
import re
from pathlib import Path

def clean_filename(filename):
    """
    Remove special characters from filename while keeping:
    - Alphanumeric characters
    - Spaces, underscores, hyphens
    - File extension
    """
    # Separate name and extension
    name, ext = os.path.splitext(filename)
    
    # Remove special characters (keep alphanumeric, spaces, underscores, hyphens)
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s_-]', '', name)
    
    # Remove multiple spaces and trim
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    
    # If name becomes empty, use a default name
    if not cleaned_name:
        cleaned_name = 'file'
    
    return cleaned_name + ext

def rename_files_in_folder(folder_path, dry_run=True):
    """
    Rename all files in folder by removing special characters.
    
    Args:
        folder_path: Path to the folder
        dry_run: If True, only show what would be changed without actually renaming
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory!")
        return
    
    # Get all files in folder
    files = [f for f in folder.iterdir() if f.is_file()]
    
    if not files:
        print("No files found in the folder.")
        return
    
    print(f"Found {len(files)} file(s) in '{folder_path}'\n")
    
    renamed_count = 0
    skipped_count = 0
    
    for file_path in files:
        old_name = file_path.name
        new_name = clean_filename(old_name)
        
        # Skip if name doesn't change
        if old_name == new_name:
            skipped_count += 1
            continue
        
        # Check for name collisions
        new_path = folder / new_name
        if new_path.exists() and new_path != file_path:
            print(f"⚠️  SKIP: '{old_name}' -> '{new_name}' (name already exists)")
            skipped_count += 1
            continue
        
        # Show what would happen
        if dry_run:
            print(f"📝 WOULD RENAME: '{old_name}' -> '{new_name}'")
        else:
            try:
                file_path.rename(new_path)
                print(f"✅ RENAMED: '{old_name}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f"❌ ERROR: '{old_name}' - {str(e)}")
    
    # Summary
    print(f"\n{'='*50}")
    if dry_run:
        print(f"DRY RUN COMPLETE - No files were actually renamed")
        print(f"Files that would be renamed: {len(files) - skipped_count}")
        print(f"Files skipped: {skipped_count}")
        print(f"\nRun with dry_run=False to actually rename files")
    else:
        print(f"RENAME COMPLETE")
        print(f"Files renamed: {renamed_count}")
        print(f"Files skipped: {skipped_count}")

# Example usage
if __name__ == "__main__":
    # Change this to your folder path
    folder_path = "/Users/harishdamodar/ingestDocs/"  # Use raw string for Windows paths
    
    # First run as dry-run to see what would change
    # rename_files_in_folder(folder_path, dry_run=True)
    
    # Then run with dry_run=False to actually rename
    rename_files_in_folder(folder_path, dry_run=False)