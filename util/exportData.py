import glob
import os
from PIL import Image, ImageOps
import shutil
from pyvox.models import Vox
from pyvox.writer import VoxWriter


def export_pngs(voxels, output_file_path, colors):
    output_file_pattern, output_file_extension = os.path.splitext(output_file_path)

    # delete the previous output files
    file_list = glob.glob(output_file_pattern + '_*.png')
    for file_path in file_list:
        try:
            os.remove(file_path)
        except Exception:
            print("Error while deleting file : ", file_path)

    z_size = voxels.shape[0]

    size = str(len(str(z_size + 1)))
    # Black background
    colors = [(0, 0, 0)] + colors
    palette = [channel for color in colors for channel in color]
    for height in range(z_size):
        # print('export png %d/%d' % (height, z_size))
        # Special case when white on black.
        if colors == [(0, 0, 0), (255, 255, 255)]:
            img = Image.fromarray(voxels[height].astype('bool'))
        else:
            img = Image.fromarray(voxels[height].astype('uint8'), mode='P')
            img.putpalette(palette)
        
        # Pillow puts (0,0) in the upper left corner, but 3D viewing coordinate systems put (0,0) in the lower left.
        # Fliping image vertically (top to bottom) makes the lower left corner (0,0) which is appropriate for this application.
        img = ImageOps.flip(img)
        path = (output_file_pattern + "_%0" + size + "d.png") % height
        img.save(path)


def exportData(padded_train_data, train_labels, exportPath, models, export_png = False, export_vox = False):
    if(export_png == False and export_vox == False):
        return

    if os.path.exists(exportPath):
        shutil.rmtree(exportPath)

    for idx, data in enumerate(padded_train_data):
        label = train_labels[idx]
        labelName = models[label]

        output_filename = f"{labelName}.png"
        png_output_path = os.path.join(exportPath,f"{idx+1}-{labelName}", output_filename)

        # make sure that the output path exists
        os.makedirs(os.path.dirname(png_output_path), exist_ok=True)

        if export_png:
            # export the pngs
            export_pngs(data, png_output_path, [(255, 255, 255)])
            
        # export the vox file
        if export_vox:
            vox = Vox.from_dense(data)
            VoxWriter(os.path.join(exportPath,f"{idx+1}-{labelName}",f"_{labelName}.vox"), vox).write()