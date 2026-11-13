import json

def patch_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('WhisperModel' in line for line in cell['source']):
            # Filter out any existing broken imports
            new_source = [
                '# BỎ QUA FILE LIBTORCHAUDIO.SO BỊ LỖI C++ TRÊN PYTHON 3.13 CỦA MOLAB\n',
                'import sys, types\n',
                'sys.modules["torchaudio"] = types.ModuleType("torchaudio")\n',
                'sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")\n',
                '\n'
            ]
            for line in cell['source']:
                if 'import torchaudio' not in line:
                    new_source.append(line)
            cell['source'] = new_source

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Patched: {path}")

patch_notebook(r'c:\Users\ACER\Downloads\notebook.ipynb')
patch_notebook(r'd:\Github\ai biet\S2ST\07_MoLab_Train_Inference_PushToHub_Master.ipynb')

