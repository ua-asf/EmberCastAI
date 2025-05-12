"""
Author: Dylan Maltos
Updated: 5/12/25
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

# Function extract_date - Extract the date from the filename
def extract_date(filename):
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
                    return dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
            # Try to parse DD_MMM(AM|PM)
            m = re.match(r'(\d{1,2})[_ ]([A-Za-z]{3})(AM|PM)?', date_str, re.IGNORECASE)
            if m:
                day = m.group(1)
                month = m.group(2)
                return f'{month}-{day}'
            return date_str  # fallback to raw
    return None

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
            # Extract date
            date = extract_date(filename)
            if not date:
                date = 'UnknownDate'
            # Extract fire name
            fire_name = extract_fire_name(filename)
            if not fire_name:
                fire_name = 'UnknownFire'
            # Create organized structure
            output_root = os.path.join(os.getcwd(), 'organized_dataset')
            fire_folder = os.path.join(output_root, fire_name, date)
            os.makedirs(fire_folder, exist_ok=True)
            # Compute hash and check for duplicates
            h = file_hash(file)
            folder_key = (fire_name, date)
            if folder_key not in folder_hashes:
                folder_hashes[folder_key] = set()
            if h in folder_hashes[folder_key]:
                continue  # Duplicate found, skip copying
            folder_hashes[folder_key].add(h)
            # Copy file to organized structure
            dst_path = os.path.join(fire_folder, filename)
            shutil.copy2(file, dst_path)