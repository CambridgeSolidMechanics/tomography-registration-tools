from pathlib import Path
import pandas as pd
import cv2 as cv
import numpy as np
from rich.progress import track

class PrintTableMetrics:
    def __init__(self, log_metrics: list, col_width: int = 10) -> None:
        super().__init__()

        header = []
        for metric in log_metrics:
            header.append(metric)
        if 'Iteration' not in header:
            header.insert(0, "Iteration")
        
        self.format_str = '{' + ':<' + str(col_width) + '}'
        self.col_width = col_width
        n_cols = len(header)
        total_width = col_width * n_cols + 3*n_cols
        self.total_width = total_width
        self.header = header
        self._time_metrics = {}
        fields = [self.format_str.format(metric) for metric in self.header]
        line = " | ".join(fields) + "\n" + "-" * self.total_width
        print(line)
    
    def update(self, metrics: dict) -> str:
        # Formatting
        s = self.format_str
        fields = []
        for key in self.header:
            if key in metrics:
                if isinstance(metrics[key], float):
                    val = f'{metrics[key]:.6f}'
                else:
                    val = metrics[key]
                fields.append(s.format(val))
            else:
                fields.append(s.format(''))
        line =  " | ".join(fields)
        print(line)

def load_images(folder: Path, image_scale_factor=4):
    ims = []
    file_list = list(folder.glob('*.tif'))
    image_indices = []
    for f in track(file_list, description='Reading images'):
        image_indices.append(int(f.stem.split('_')[-1]))
        im = cv.imread(str(f), cv.IMREAD_UNCHANGED)
        dtype = im.dtype
        im = cv.resize(im, None, fx=1/image_scale_factor, fy=1/image_scale_factor)
        im = 1 - im.astype(np.float32)/np.iinfo(dtype).max
        ims.append(im)
    proj_data = np.stack(ims, axis=1)
    del ims
    return proj_data, image_indices

def load_xtekct(fn: Path):
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
    else:
        pth = next(fn.glob('*.xtekct'))
    assert pth.exists()
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    data = {}

    for line in lines:
        if ('[' in line) and (']' in line):
            current = line.strip('[]')
            data[current] = {}
        elif len(line)<1:
            pass
        else:
            fields = line.split('=')
            key = fields[0]
            value = '='.join(fields[1:])
            try:
                value = float(value)
            except ValueError:
                pass
            data[current][key] = value
            
    return data

def load_from_ang(pth: Path):
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    # skip 1st line and load from 2nd with delimiter ':'
    data = {}
    for line in lines[1:]:
        if len(line)<1: continue
        key, val = line.split(':')
        data[int(key)] = float(val)
    return data

def load_from_ctdata(pth: Path):
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    for i, line in enumerate(lines):
        if 'Angle(deg)' in line:
            break
    df = pd.read_csv(pth, skiprows=i, sep='\s+')
    indices = df['Projection'].values
    angles = df['Angle(deg)'].values
    return dict(zip(indices, angles))

def load_angles(fn: Path):
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
        assert pth.exists()
    else:
        files = list(fn.glob('*.ang')) + list(fn.glob('*_ctdata*'))
        assert len(files) == 1
        pth = files[0]
    if 'ang' in pth.suffix:
        return load_from_ang(pth)
    elif 'ctdata' in pth.stem:
        return load_from_ctdata(pth)