from PIL import Image
import os

folder_path_input = "C:/Users/H213561/Documents/magnet/in-words-ashu/"
folder_path_output= "C:/Users/H213561/Documents/magnet/in-words-ashu-transparentBG/"

image_names =os.listdir(folder_path_input)

for img_name in image_names:
    print img_name
    img = Image.open(folder_path_input+img_name)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    #os.chdir(folder_path_output)
    img.putdata(newData)
    img.save(folder_path_output+img_name, "PNG")
