import os
import requests
from zipfile import ZipFile
from io import BytesIO

# Base URL
base_url = "https://s3.amazonaws.com/tripdata/"

# Correct months list
months = [
    "JC-202401", "JC-202402", "JC-202403", "JC-202404",
    "JC-202405", "JC-202406", "JC-202407", "JC-202408",
    "JC-202409", "JC-202410", "JC-202411", "JC-202412"
]

# SAFELY Create 'data/raw' folder
if not os.path.exists('data/raw'):
    os.makedirs('data/raw')
elif not os.path.isdir('data/raw'):
    raise NotADirectoryError("'data/raw' exists but is not a directory!")

# Now your download loop here
for month in months:
    filename = f"{month}-citibike-tripdata.csv.zip"
    url = base_url + filename
    print(f"🚴 Downloading {url}...")

    response = requests.get(url)
    if response.status_code == 200:
        print(f"📦 Extracting {filename}...")
        with ZipFile(BytesIO(response.content)) as zip_file:
            for member in zip_file.namelist():
                if member.endswith('/'):
                    continue
                extracted_filename = os.path.basename(member)
                save_path = os.path.join('data', 'raw', extracted_filename)

                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with zip_file.open(member) as source_file, open(save_path, "wb") as target_file:
                    target_file.write(source_file.read())
        print(f"✅ {filename} extracted successfully!\n")
    else:
        print(f"❌ Failed to download {filename} - Status Code: {response.status_code}")

print("✨ All downloads and extractions complete!")
