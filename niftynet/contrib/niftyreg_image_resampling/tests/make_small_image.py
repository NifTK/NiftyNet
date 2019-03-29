import nibabel as nib
import numpy as np
import argparse as ap

parser = ap.ArgumentParser(description='Image gen')

parser.add_argument('destination')
parser.add_argument('-x', '--x', type=int, default=100)
parser.add_argument('-y', '--y', type=int, default=110)
parser.add_argument('-z', '--z', type=int, default=1)

args = parser.parse_args()

data = np.zeros((args.x, args.y, args.z))
for x in range(args.x):
    for y in range(args.y):
        for z in range(args.z):
            data[x,y,z] = x + 2*y + 3*z

if args.z == 1:
    data = data.reshape((args.x, args.y))

nib.save(nib.Nifti1Image(data, np.eye(4)), args.destination)
