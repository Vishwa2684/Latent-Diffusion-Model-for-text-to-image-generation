import requests
import zipfile
import os

def download_and_extract_zip(url, output_directory):
    """
    Downloads a ZIP file from a URL, extracts its contents, and saves them to an output directory.

    :param url: URL of the ZIP file to download.
    :param output_directory: Path to the directory where the extracted files will be saved.
    """
    zip_filename = "temp_download.zip"

    try:
        # Download the ZIP file
        print(f"Downloading ZIP file from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError if the request fails

        # Save the ZIP file locally
        with open(zip_filename, "wb") as zip_file:
            for chunk in response.iter_content(chunk_size=8192):
                zip_file.write(chunk)
        print("Download complete.")

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Extract the ZIP file
        print(f"Extracting contents to {output_directory}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
        print("Extraction complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary ZIP file
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            print("Temporary file cleaned up.")

# Example usage
url = "http://images.cocodataset.org/zips/train2014.zip"
output_directory = "/teamspace/studios/this_studio/coco"  # Replace with your desired output directory

download_and_extract_zip(url, output_directory)
