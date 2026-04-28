# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
E2V Demo Script
"""

import numpy as np
import argparse
import os

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='', help='path to reconstructed event file')
    params, _ = parser.parse_known_args(argv)
    return params

def main():
    params = parse_args()
    params.out_file = params.file[:-4] + "_formatted.mp4"
    os.system(f"ffmpeg -i {params.file} -ss 00:00:02.500 -c:v libx264 -pix_fmt yuv420p -crf 18 {params.out_file}")

if __name__ == '__main__':
    main()
