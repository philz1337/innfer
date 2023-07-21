import os
import requests

def download_file(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        progress = 0

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress += len(chunk)
                percentage = (progress / total_size) * 100
                print(f"Downloaded: {progress}/{total_size} bytes ({percentage:.2f}%)", end='\r')
        print("\nDownload complete!")

def main():
    url = "https://huggingface.co/gemasai/8x_NMKD-Faces_160000_G/resolve/main/8x_NMKD-Faces_160000_G.pth"
    save_path = "./models/8x_NMKD-Faces_160000_G.pth"

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        download_file(url, save_path)
    else:
        print("File already exists. Skipping download.")

if __name__ == "__main__":
    main()
