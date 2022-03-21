"""
The MIT License (MIT)

Copyright (c) 2015 Daniel Maturana, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Read and write PCL .pcd files in python.
dimatura@cmu.edu, 2013
"""

import re
import struct
import copy
import numpy as np

from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2

# -----------------------------------------------------------------------------
# ----- Settings
# -----------------------------------------------------------------------------

DUMMY_FIELD_PREFIX = '__'

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

pc2_pcd_type_mappings = [(PointField.INT8, ('I', 1)),
                         (PointField.UINT8, ('U', 1)),
                         (PointField.INT16, ('I', 2)),
                         (PointField.UINT16, ('U', 2)),
                         (PointField.INT32, ('I', 4)),
                         (PointField.UINT32, ('U', 4)),
                         (PointField.FLOAT32, ('F', 4)),
                         (PointField.FLOAT64, ('F', 8))]
pc2_type_to_pcd_type = dict(pc2_pcd_type_mappings)
pcd_type_to_pc2_type = dict((q, p) for (p, q) in pc2_pcd_type_mappings)

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


# -----------------------------------------------------------------------------
# ----- numpy <-> pc2 bridge
# -----------------------------------------------------------------------------

def split_rgb_field(cloud_arr):
    '''Takes an array with a named 'rgb' float32 field, and returns an array in which
    this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.
    (pcl stores rgb in packed 32 bit floats)
    '''
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgb':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr


def merge_rgb_fields(cloud_arr):
    '''Takes an array with named np.uint8 fields 'r', 'g', and 'b', and returns an array in
    which they have been merged into a single np.float32 'rgb' field. The first byte of this
    field is the 'r' uint8, the second is the 'g', uint8, and the third is the 'b' uint8.
    This is the way that pcl likes to handle RGB colors for some reason.
    '''
    r = np.asarray(cloud_arr['r'], dtype=np.uint32)
    g = np.asarray(cloud_arr['g'], dtype=np.uint32)
    b = np.asarray(cloud_arr['b'], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of the array
    # from uint32 to float32, but i don't want any conversion to take place -jdb
    rgb_arr.dtype = np.float32

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ('r', 'g', 'b'):
            new_dtype.append((field_name, field_type))
    new_dtype.append(('rgb', np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'rgb':
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def arr_to_fields(cloud_arr):
    '''Convert a numpy record datatype into a list of PointFields.
    '''
    fields = []
    for field_name in cloud_arr.dtype.names:
        np_field_type, field_offset = cloud_arr.dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        pf.count = 1  # is this ever more than one?
        fields.append(pf)
    return fields


def pointcloud2_to_dtype(cloud_msg):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in cloud_msg.fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
        np_dtype_list.append((f.name, pftype_to_nptype[f.datatype]))
        offset += pftype_sizes[f.datatype]

    # might be extra padding between points
    while offset < cloud_msg.point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def pointcloud2_to_array(cloud_msg, split_rgb=False, remove_padding=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = pointcloud2_to_dtype(cloud_msg)

    # parse the cloud into an array
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    if remove_padding:
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if split_rgb:
        cloud_arr = split_rgb_field(cloud_arr)

    return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None, merge_rgb=False):
    '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
    '''
    if merge_rgb:
        cloud_arr = merge_rgb_fields(cloud_arr)

    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)

    cloud_msg = PointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = arr_to_fields(cloud_arr)
    cloud_msg.is_bigendian = False  # assumption
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * cloud_arr.shape[1]
    cloud_msg.is_dense = all([np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])
    cloud_msg.data = cloud_arr.tostring()
    return cloud_msg


# -----------------------------------------------------------------------------
# ----- Read/Write routines
# -----------------------------------------------------------------------------

def parse_header(lines):
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            print("\033[93m" + "warning: can't understand line: %s" % ln + "\033[1m")
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = list(map(int, value.split()))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = list(map(float, value.split()))
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def write_header(metadata, rename_padding=False):
    """ given metadata as dictionary return a string header.
    """
    template = """\
VERSION {version}
FIELDS {fields}
SIZE {size}
TYPE {type}
COUNT {count}
WIDTH {width}
HEIGHT {height}
VIEWPOINT {viewpoint}
POINTS {points}
DATA {data}
"""
    str_metadata = metadata.copy()

    if not rename_padding:
        str_metadata['fields'] = ' '.join(metadata['fields'])
    else:
        new_fields = []
        for f in metadata['fields']:
            if f == '_':
                new_fields.append('padding')
            else:
                new_fields.append(f)
        str_metadata['fields'] = ' '.join(new_fields)
    str_metadata['size'] = ' '.join(map(str, metadata['size']))
    str_metadata['type'] = ' '.join(metadata['type'])
    str_metadata['count'] = ' '.join(map(str, metadata['count']))
    str_metadata['width'] = str(metadata['width'])
    str_metadata['height'] = str(metadata['height'])
    str_metadata['viewpoint'] = ' '.join(map(str, metadata['viewpoint']))
    str_metadata['points'] = str(metadata['points'])
    tmpl = template.format(**str_metadata)
    return tmpl


def _metadata_is_consistent(metadata):
    """ sanity check for metadata. just some basic checks.
    """
    checks = []
    required = ('version', 'fields', 'size', 'width', 'height', 'points',
                'viewpoint', 'data')
    for f in required:
        if f not in metadata:
            print('%s required' % f)
    checks.append((lambda m: all([k in m for k in required]),
                   'missing field'))
    checks.append((lambda m: len(m['type']) == len(m['count']) ==
                             len(m['fields']),
                   'length of type, count and fields must be equal'))
    checks.append((lambda m: m['height'] > 0,
                   'height must be greater than 0'))
    checks.append((lambda m: m['width'] > 0,
                   'width must be greater than 0'))
    checks.append((lambda m: m['points'] > 0,
                   'points must be greater than 0'))
    checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                                                   'binary_compressed'),
                   'unknown data type:'
                   'should be ascii/binary/binary_compressed'))
    ok = True
    for check, msg in checks:
        if not check(metadata):
            print('error:', msg)
            ok = False
    return ok


def _build_dtype(metadata):
    """ build numpy structured array dtype from pcl metadata.
    note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.
    TODO: allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points'] * dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def point_cloud_from_fileobj(f):
    """ parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip()
        if not isinstance(ln, str):
            ln = ln.decode('utf-8')
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break

    pc_data = parse_binary_pc_data(f, dtype, metadata)
    return PointCloud(metadata, pc_data)


def point_cloud_from_path(fname):
    """ load point cloud in binary format
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc


def point_cloud_to_fileobj(pc, fileobj, data_compression=None):
    """ write pointcloud as .pcd to fileobj.
    if data_compression is not None it overrides pc.data.
    """
    metadata = pc.get_metadata()
    if data_compression is not None:
        data_compression = data_compression.lower()
        assert (data_compression in ('ascii', 'binary', 'binary_compressed'))
        metadata['data'] = data_compression

    header = write_header(metadata).encode('utf-8')
    fileobj.write(header)
    fileobj.write(pc.pc_data.tostring())


class PointCloud(object):
    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        # pdb.set_trace()
        md = self.get_metadata()
        assert (_metadata_is_consistent(md))
        assert (len(self.pc_data) == self.points)
        assert (self.width * self.height == self.points)
        assert (len(self.fields) == len(self.count))
        assert (len(self.fields) == len(self.type))

    def save_pcd(self, fname, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            print('\033[93m' + 'data_compression keyword is deprecated for'
                               ' compression' + '\033[1m')
            compression = kwargs['data_compression']
        with open(fname, 'wb') as f:
            point_cloud_to_fileobj(self, f, compression)

    def save_pcd_to_fileobj(self, fileobj, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            print('\033[93m' + 'data_compression keyword is deprecated for'
                               ' compression' + '\033[1m')
            compression = kwargs['data_compression']
        point_cloud_to_fileobj(self, fileobj, compression)

    def copy(self):
        new_pc_data = np.copy(self.pc_data)
        new_metadata = self.get_metadata()
        return PointCloud(new_metadata, new_pc_data)

    def to_msg(self):
        # TODO is there some metadata we want to attach?
        return array_to_pointcloud2(self.pc_data)

    @staticmethod
    def from_path(fname):
        return point_cloud_from_path(fname)

    @staticmethod
    def from_msg(msg, squeeze=True):
        """ from pointcloud2 msg
        squeeze: fix when clouds get 1 as first dim
        """
        md = {'version': .7,
              'fields': [],
              'size': [],
              'count': [],
              'width': 0,
              'height': 1,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': 0,
              'type': [],
              'data': 'binary_compressed'}
        for field in msg.fields:
            md['fields'].append(field.name)
            t, s = pc2_type_to_pcd_type[field.datatype]
            md['type'].append(t)
            md['size'].append(s)
            # TODO handle multicount correctly
            if field.count > 1:
                print('\033[93m' + 'fields with count > 1 are not well tested' + '\033[1m')
            md['count'].append(field.count)
        pc_data = np.squeeze(pointcloud2_to_array(msg))
        md['width'] = len(pc_data)
        md['points'] = len(pc_data)
        pc = PointCloud(md, pc_data)
        return pc
