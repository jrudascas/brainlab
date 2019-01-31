import os

folder_path = '/home/brainlab/Desktop/Rudas/Data/Lorina_Propofol_Resting_State/'

cont = 1
for (root, dirs, files) in os.walk(folder_path):
    for file in sorted(files):
        if not file.endswith('.nii'):
            try:
                os.remove(root + '/' + file)
            except Exception:
                continue
        else:
            if not file.startswith('f') and not file.startswith('s'):
                try:
                    os.remove(root + '/' + file)
                except Exception:
                    continue