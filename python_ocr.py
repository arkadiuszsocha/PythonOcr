import os
import sys
import easyocr
import ssl
import certifi
import cv2
import numpy as np

# Set SSL certificate path
ssl._create_default_https_context = ssl._create_unverified_context

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize the image
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return resized

def process_folder_with_easyocr(folder_path):
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['pl'], gpu=False)  # Add more languages if needed

    # Verify the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        sys.exit(1)

    # Create output folder for text files
    output_folder = os.path.join(folder_path, "OCR_Results")
    os.makedirs(output_folder, exist_ok=True)

    # Process all image files in the folder
    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            # Preprocess the image
            try:
                preprocessed_image = preprocess_image(file_path)
                # Perform OCR on the preprocessed image
                results = reader.readtext(
                    preprocessed_image, 
                    detail=1, 
                    paragraph=True, 
                    text_threshold=0.7, 
                    low_text=0.4, 
                    link_threshold=0.4,
                    x_ths=1.0,  # Adjust X-axis threshold
                    y_ths=0.5,  # Adjust Y-axis threshold
                    min_size=20,  # Minimum size of text to detect
                    add_margin=0.1  # Add margin around detected text
                )
                extracted_text = "\n".join([res[1] for res in results])  # Extract text only

                # Save the text to a file
                text_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(extracted_text)

                print(f"Text extracted and saved to: {text_file_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nOCR processing completed. Results saved in: {output_folder}")

if __name__ == "__main__":
    # Check if folder path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    process_folder_with_easyocr(folder_path)



    # running 
    #python easyocr_console_processor.py /path/to/your/folder
    #/Users/arek/Documents/repos/PythonOcr/images
    # python3 python_ocr.py /Users/arek/Documents/repos/PythonOcr/images
# ssl pip3 install certifi