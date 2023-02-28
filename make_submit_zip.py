import os
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default='./results/test_warped_model',
    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='./submit',
    type=str, help='Directory for output')
parser.add_argument('--file_name', default='warped_model',
    type=str, help='Directory for output')
parser.add_argument('--runtime', default=1.69, type=float, help='runtime per image')
parser.add_argument('--ex_data', default=0, type=int, 
    choices=[0, 1], help='whether to include extra data [1 / 0]')
parser.add_argument('--other_desc', default="", type=str, help='other description')
args = parser.parse_args()

readme_content_list = [
    f'runtime per image [s] : {args.runtime}',
    f'CPU[1] / GPU[0] : 0',
    f'Extra Data [1] / No Extra Data [0] : {args.ex_data}',
    f'Other description : {args.other_desc}'
]


tmp_dir = Path(args.output_dir) / 'tmp'
os.makedirs(tmp_dir, exist_ok=True)
shutil.rmtree(tmp_dir)
shutil.copytree(args.result_dir, tmp_dir)

with open(tmp_dir / 'readme.txt', 'w') as f:
    f.write('\n'.join(readme_content_list))

shutil.make_archive(Path(args.output_dir) / args.file_name, format='zip', root_dir=tmp_dir)