def get_latest(files):
    return files[-1]

def change_to_list(files):
    if not isinstance(files, list):
        list_mask = [files]
    else:
        list_mask=files

    return list_mask

def get_wm_csf(files):
    print(len(files))
    wm_csf_list = [files[0], files[2]]
    return wm_csf_list