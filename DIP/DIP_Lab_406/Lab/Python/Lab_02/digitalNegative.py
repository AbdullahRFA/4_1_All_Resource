from PIL import Image, ImageOps
image_path = 'Image/nature.jpeg'
image = Image.open(image_path)
negative_image = ImageOps.invert(image)
negative_image.save('negative_image.jpg')
negative_image.show()
