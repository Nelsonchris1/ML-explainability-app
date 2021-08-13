import os
import stat


def remove_files():
    files = ['model2', 'img_pdplot.png', 'shapvalue.png']

    for file in files:
        if os.path.exists(file):
            os.remove(file)
        else:
            pass