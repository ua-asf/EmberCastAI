"""
Author: Dylan Maltos
Updated: 5/14/25
preprocess.py - Preprocesses the dataset by extracting the fire name and date from the filename and organizing the files into folders.
"""

# Libraries
import os
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import shutil
import re
import hashlib

# Regex patterns for various date/time formats
DATE_PATTERNS = [
    (r'(\d{8})', '%Y%m%d'),  # 20210905
    (r'(\d{8}_[0-9]{4,6}(PDT|AM|PM)?)', '%Y%m%d'),  # 20210903_122318PDT
    (r'(\d{1,2}_[A-Za-z]{3}(AM|PM)?)', None),  # 12_NovPM
    (r'(\d{1,2} [A-Za-z]{3}( AM| PM)?)', None),  # 12 Nov AM
    (r'(\d{4}(PDT|AM|PM))', None),  # 0030PDT, 0415AM
]

IGNORED_NAMES = {'AM', 'PM', 'IR', 'Div', 'HeatPerimeter', 'per', 'Complex', 'Photo'}

# Function extract_date_time - Extract the date and time from the filename
# Returns (date, time) separately
def extract_date_time(filename):
    date = None
    time = None
    # First, extract date
    for pattern, date_fmt in DATE_PATTERNS:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Handle "YYYYMMDD_HHMM" by extracting just the date
            if '_' in date_str:
                date_str = date_str.split('_')[0]
            if date_fmt:
                try:
                    dt = datetime.strptime(date_str, date_fmt)
                    date = dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
            # Try to parse DD_MMM(AM|PM)
            m = re.match(r'(\d{1,2})[_ ]([A-Za-z]{3})(AM|PM)?', date_str, re.IGNORECASE)
            if m:
                day = m.group(1)
                month = m.group(2)
                date = f'{month}-{day}'
            if not date and date_str:
                date = date_str  # fallback to raw
            break
    # Now, extract time (look for HHMM or HHMMSS, possibly with PDT/AM/PM)
    # Only extract time from the part after the date
    time = None
    time_match = None
    if date:
        # Find the position of the date in the filename
        idx = filename.find(date_str)
        after_date = filename[idx+len(date_str):] if idx != -1 else filename
        time_match = re.search(r'([01]\d|2[0-3])[0-5]\d', after_date)
    if time_match:
        time = time_match.group(0)[:4]  # Only HHMM
    if not date:
        date = 'UnknownDate'
    if not time:
        time = '0000'
    return date, time

# Function extract_fire_name - Extract the fire name from the filename
def extract_fire_name(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Remove all date/time patterns
    for pattern, _ in DATE_PATTERNS:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    # Remove leading/trailing underscores, dashes, spaces
    name = name.strip('_- .')
    # Remove known suffixes
    name = re.sub(r'(_IR|_Div|_HeatPerimeter|_per|_Photo\d+WithFirePerimeter|_Complex)?$', '', name)
    # Split by underscores, dashes, or spaces
    parts = re.split(r'[_\-\s]+', name)
    # Find the first part that is not empty, not a time, and not in IGNORED_NAMES
    for part in parts:
        if not part:
            continue
        # Ignore if it's a time (e.g., 0030PDT, 122318PDT, etc.)
        if re.match(r'^(\d{3,6}(PDT|AM|PM)?)$', part, re.IGNORECASE):
            continue
        if part.upper() in IGNORED_NAMES:
            continue
        # If it's a valid fire name 
        if len(part) >= 3:
            return part
    return 'UnknownFire'

def file_hash(filepath):
    """Compute SHA256 hash of a file's contents."""
    hash_func = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()

# Dictionary to store hashes for each fire/date folder
folder_hashes = {}

for dirpath, dirnames, filenames in os.walk('dataset'):
    for filename in filenames:
        if filename.endswith(('.png', '.wkt', '.tif')):
            file = os.path.join(dirpath, filename)
            # Skip any 'squares' folders
            if 'squares' in os.path.normpath(file).split(os.sep):
                continue
            # Extract date and time
            date, time = extract_date_time(filename)
            folder_name = f"{date}-{time}"
            # Extract fire name
            fire_name = extract_fire_name(filename)
            if not fire_name:
                fire_name = 'UnknownFire'
            # Create organized structure
            output_root = os.path.join(os.getcwd(), 'organized_dataset')
            fire_folder = os.path.join(output_root, fire_name, folder_name)
            os.makedirs(fire_folder, exist_ok=True)
            # Compute hash and check for duplicates
            h = file_hash(file)
            folder_key = (fire_name, folder_name)
            if folder_key not in folder_hashes:
                folder_hashes[folder_key] = set()
            if h in folder_hashes[folder_key]:
                continue  # Duplicate found, skip copying
            folder_hashes[folder_key].add(h)
            # Copy file to organized structure
            dst_path = os.path.join(fire_folder, filename)
            shutil.copy2(file, dst_path)