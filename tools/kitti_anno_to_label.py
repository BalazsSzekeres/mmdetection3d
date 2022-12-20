import argparse
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import pickle


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', None),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


def kitti_anno_to_label_file(annos, output_folder):
    output_folder = Path(output_folder)

    for anno in tqdm(annos):
        image_idx = anno['frame_id']
        label_lines = []
        for j in range(anno["bbox"].shape[0]):
            label_dict = {
                'name': anno["name"][j],
                'alpha': anno["alpha"][j],
                'bbox': anno["bbox"][j],
                'location': anno["location"][j],
                'dimensions': anno["dimensions"][j][[1, 2, 0]],  # kitti dimensions are in a weird order
                'rotation_y': anno["rotation_y"][j],
                'score': anno["score"][j],
            }
            label_line = kitti_result_line(label_dict)
            # if anno["score"][j] > 0.3:
            label_lines.append(label_line)
        label_file = output_folder / f"{image_idx}.txt"
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generates labels  from pickle files.')
    parser.add_argument('annos', help='pickle file')
    parser.add_argument('output_folder', help='Output folder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    detections = pickle.load(open(args.annos, 'rb'))
    kitti_anno_to_label_file(detections, args.output_folder)


if __name__ == '__main__':
    main()
