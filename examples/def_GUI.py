# STILL BEING DEVELOPED
# A GUI to load volumes and apply a uniform or gaussian deformation to it.

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
# if os.path.abspath('../../tomography-registration-tools') not in sys.path:
#     sys.path.insert(0, os.path.abspath('../../tomography-registration-tools'))
if os.path.abspath('../tomography-registration-tools') not in sys.path:
    sys.path.insert(0, os.path.abspath('../tomography-registration-tools'))
from pathlib import Path
import numpy as np
import yaml

from t_core import volume_tools as vt
from t_core import fields


def applyDef(volFileName,
             defName,
             defType,
             eps_xyz=None,
             offset_xyz=None,
             sigma_vox=None,
             A_vox=None):
    
    # volFileName = askopenfilename(initialdir='.', title='Volume to be deformed')
    vol = vt.load_volume(
        volFileName,
        resolution=resolution,
    )
    update_log('Loaded volume')
    for x,y in zip(resolution, vol.shape):
        assert x==y
    if defType == 'Uniform':
        f = fields.UniformStrainDisplacementField(eps_xyz=eps_xyz, offset_xyz=offset_xyz)
    elif defType == 'Gaussian':
        pos = [[r//2 for r in resolution]]
    
        L_x = resolution[0] / 2
        L_y = resolution[1] / 2
        L_z = resolution[2] / 2
        for i in range(len(pos)):
            c_x, c_y, c_z = pos[i]
            c_x = c_x / L_x - 1
            c_y = c_y / L_y - 1
            c_z = c_z / L_z - 1
            pos[i] = (c_x, c_y, c_z)
            
        s_x = sigma_vox / L_x
        s_y = sigma_vox / L_y
        s_z = sigma_vox / L_z
        A_z = A_vox / L_z
        
        f = fields.AdditiveFieldArray([ fields.GaussianDisplacementField((s_x,s_y,s_z), (0,0,A_z), p) for p in pos ])
        
    f.to_yaml(f'field_{defName}.yaml')
    undef_vol = f(vol)
    update_log('Applied displacement field')
    fold = Path('./')
    
    # fn = asksaveasfilename(initialdir='.', title='deformed volume file', defaultextension='.raw')
    fn = volFileName[:-4]+'_'+defName+'.raw'
    vt.save_volume(fn, undef_vol)
    (fold / Path(f'resolution_{defName}.txt')).write_text('x'.join(map(str, vol.shape)))


def update_log(message):
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)
    root.update()

def submit():
    status.config(text = 'Busy', bg='red')
    volume_path = volume_entry.get()
    deformation_name = name_entry.get()
    deformation_type = type_var.get()

    # Check if file with deformation name exists
    if os.path.exists(f"field_{deformation_name}.yaml"):
        if not messagebox.askyesno("File Exists", "A file with the same name already exists. Do you want to continue with this name?"):
            return

    update_log(f"Volume to Deform: {volume_path}")
    update_log(f"Deformation Name: {deformation_name}")
    update_log(f"Deformation Type: {deformation_type}")

    if deformation_type == "Uniform":
        eps_x = eps_x_entry.get()
        eps_y = eps_y_entry.get()
        eps_z = eps_z_entry.get()
        eps_xyz = (float(eps_x), float(eps_y), float(eps_z))

        offset_x = offset_x_entry.get()
        offset_y = offset_y_entry.get()
        offset_z = offset_z_entry.get()
        offset_xyz = (float(offset_x), float(offset_y), float(offset_z))

        update_log(f"Epsilon_xyz: {(eps_x, eps_y, eps_z)}")
        update_log(f"Offset_xyz: {(offset_x, offset_y, offset_z)}")

        applyDef(volFileName=volume_path,
                defName=deformation_name,
                defType=deformation_type,
                eps_xyz=eps_xyz,
                offset_xyz=offset_xyz)

    elif deformation_type == "Gaussian":
        sig_vox = int(sig_entry.get())
        amp_z = int(amp_entry.get())
        update_log(f"Sigma (vox): {sig_vox}")
        update_log(f"Amplitude (z, vox): {amp_z}")
    
        applyDef(volFileName=volume_path,
                defName=deformation_name,
                defType=deformation_type,
                sigma_vox=sig_vox,
                A_vox=amp_z)
        
    status.config(text = 'open to inputs', bg='green')

def browse_volume():
    volume_path = filedialog.askopenfilename()
    volume_entry.delete(0, tk.END)
    volume_entry.insert(0, volume_path)

if __name__=='__main__':
    resolution = [int(x) for x in Path('./resolution.txt').read_text().split('\n')[0].split('x')]
    
    root = tk.Tk()
    root.title("Deformation Parameters")

    # Volume to deform
    volume_label = tk.Label(root, text="Volume to Deform:")
    volume_label.grid(row=0, column=0, sticky="w")

    volume_entry = tk.Entry(root, width=50)
    volume_entry.insert(0, 'vol1.raw')
    volume_entry.grid(row=0, column=1, padx=10, pady=5)

    browse_button = tk.Button(root, text="Browse", command=browse_volume)
    browse_button.grid(row=0, column=2, padx=5, pady=5)

    # Deformation name
    name_label = tk.Label(root, text="Deformation Name:")
    name_label.grid(row=1, column=0, sticky="w")

    name_entry = tk.Entry(root, width=50)
    name_entry.insert(0, 'def1')
    name_entry.grid(row=1, column=1, padx=10, pady=5)

    # Deformation type
    type_label = tk.Label(root, text="Deformation Type:")
    type_label.grid(row=2, column=0, sticky="w")

    type_var = tk.StringVar(root)
    type_var.set("Uniform")  # Default value

    type_dropdown = tk.OptionMenu(root, type_var, "Uniform", "Gaussian")
    type_dropdown.grid(row=2, column=1, padx=10, pady=5)

    # Epsilon x (Uniform Deformation)
    eps_x_label = tk.Label(root, text="Epsilon_x:")
    eps_x_label.grid(row=3, column=0, sticky="w")

    eps_x_entry = tk.Entry(root, width=50)
    eps_x_entry.insert(0, '0.0')
    eps_x_entry.grid(row=3, column=1, padx=10, pady=5)

    # Epsilon y (Uniform Deformation)
    eps_y_label = tk.Label(root, text="Epsilon_y:")
    eps_y_label.grid(row=4, column=0, sticky="w")

    eps_y_entry = tk.Entry(root, width=50)
    eps_y_entry.insert(0, '0.0')
    eps_y_entry.grid(row=4, column=1, padx=10, pady=5)

    # Epsilon z (Uniform Deformation)
    eps_z_label = tk.Label(root, text="Epsilon_z:")
    eps_z_label.grid(row=5, column=0, sticky="w")

    eps_z_entry = tk.Entry(root, width=50)
    eps_z_entry.insert(0, '0.0')
    eps_z_entry.grid(row=5, column=1, padx=10, pady=5)

    # Offset x (Uniform Deformation)
    offset_x_label = tk.Label(root, text="Offset_x:")
    offset_x_label.grid(row=3, column=2, sticky="w")

    offset_x_entry = tk.Entry(root, width=50)
    offset_x_entry.insert(0, '0.0')
    offset_x_entry.grid(row=3, column=3, padx=10, pady=5)

    # Offset y (Uniform Deformation)
    offset_y_label = tk.Label(root, text="Offset_y:")
    offset_y_label.grid(row=4, column=2, sticky="w")

    offset_y_entry = tk.Entry(root, width=50)
    offset_y_entry.insert(0, '0.0')
    offset_y_entry.grid(row=4, column=3, padx=10, pady=5)

    # Offset z (Uniform Deformation)
    offset_z_label = tk.Label(root, text="Offset_z:")
    offset_z_label.grid(row=5, column=2, sticky="w")

    offset_z_entry = tk.Entry(root, width=50)
    offset_z_entry.insert(0, '0.0')
    offset_z_entry.grid(row=5, column=3, padx=10, pady=5)

    # Gaussian Deformation Fields (Hidden by Default)
    sig_label = tk.Label(root, text="Sigma:")
    sig_label.grid(row=3, column=0, sticky="w")
    sig_label.grid_remove()

    sig_entry = tk.Entry(root, width=50)
    sig_entry.insert(0, '125')
    sig_entry.grid(row=3, column=1, padx=10, pady=5)
    sig_entry.grid_remove()

    amp_label = tk.Label(root, text="Amplitude:")
    amp_label.grid(row=4, column=0, sticky="w")
    amp_label.grid_remove()

    amp_entry = tk.Entry(root, width=50)
    amp_entry.insert(0, '30')
    amp_entry.grid(row=4, column=1, padx=10, pady=5)
    amp_entry.grid_remove()

    # Function to toggle visibility of Gaussian Deformation Fields
    def toggle_fields():
        if type_var.get() == "Uniform":
            eps_x_label.grid()
            eps_x_entry.grid()
            eps_y_label.grid()
            eps_y_entry.grid()
            eps_z_label.grid()
            eps_z_entry.grid()
            offset_x_label.grid()
            offset_x_entry.grid()
            offset_y_label.grid()
            offset_y_entry.grid()
            offset_z_label.grid()
            offset_z_entry.grid()
            sig_label.grid_remove()
            sig_entry.grid_remove()
            amp_label.grid_remove()
            amp_entry.grid_remove()
        elif type_var.get() == "Gaussian":
            eps_x_label.grid_remove()
            eps_x_entry.grid_remove()
            eps_y_label.grid_remove()
            eps_y_entry.grid_remove()
            eps_z_label.grid_remove()
            eps_z_entry.grid_remove()
            offset_x_label.grid_remove()
            offset_x_entry.grid_remove()
            offset_y_label.grid_remove()
            offset_y_entry.grid_remove()
            offset_z_label.grid_remove()
            offset_z_entry.grid_remove()
            sig_label.grid()
            sig_entry.grid()
            amp_label.grid()
            amp_entry.grid()

    type_var.trace_add("write", lambda *args: toggle_fields())

    status = tk.Label(root, text="Open to inputs", bg='green')
    status.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

    # Submit button
    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.grid(row=6, column=1, pady=10)

    # Log Window
    log_frame = tk.Frame(root)
    log_frame.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

    log_label = tk.Label(log_frame, text="Log:")
    log_label.pack(side=tk.TOP, anchor="w")

    log_text = tk.Text(log_frame, height=10, width=60)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    log_scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    log_text.config(yscrollcommand=log_scrollbar.set)
    update_log('Resolution: '+str(resolution)+'\n')

    root.mainloop()