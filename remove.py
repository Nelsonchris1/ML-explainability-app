import os
import stat


def remove_files():
    files = ['model2', 'img_pdplot.png', 'shapvalue.png', 'lime.html']

    for file in files:
        if os.path.exists(file):
            os.remove(file)
        else:
            pass