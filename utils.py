import os
from datetime import datetime
import shutil
date_format_str = '%Y-%m-%d-%H%M'

def get_fires() -> dict:
    # Fires is a dict of each fire, with each fire being a 
    # list of the files associated with that fire, sorted
    # by date
    fires = dict()

    # Grab all fires (they are organized as organized_dataset/<FIRENAME>/<FILES>)
    dirnames = os.listdir('organized_dataset')
    for dirname in dirnames:
        # Check if the directory is a fire
        if os.path.isdir(os.path.join('organized_dataset', dirname)):
            try:
                # Get all folder in the directory
                dates = os.listdir(os.path.join('organized_dataset', dirname))
                # Purge any 'unknown date' folders
                dates = [date for date in dates if date != 'UnknownDate' and date != 'data']
                # Sort the files by date
                dates.sort(key=lambda x: datetime.strptime(x, date_format_str))
                # Skip any fires with less than 2 days of data
                if len(dates) < 2:
                    print(f'Skipping {dirname} with {len(dates)} days of data')
                    continue
                # Add the fire to the fires dict
                fires[dirname] = [os.path.join('organized_dataset', dirname, file) for file in dates]
            except Exception as e:
                print(f'Error with {dirname}: {e}')
                continue
            

    # Purge any fires not in the fires dict
    for fire in dirnames:
        if fire not in fires:
            # Check if the directory is a fire
            if os.path.isdir(os.path.join('organized_dataset', fire)):
                # Remove the directory
                shutil.rmtree(f'organized_dataset/{fire}')
                print(f'Removed {fire}')

    return fires