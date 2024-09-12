import pytesseract
from PIL import Image

print(pytesseract.image_to_string(Image.open(r'C:\Users\janha\OneDrive\Pictures\quotes4.jpg')))