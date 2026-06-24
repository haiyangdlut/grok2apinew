import struct

with open('/home/ubuntu/kg/data/files/videos/video_b1764d89636145748ead924b95306ede.mp4', 'rb') as f:
    data = f.read()

idx = data.find(b'tkhd')
if idx < 0:
    print('tkhd not found')
else:
    offset = idx - 4
    version = data[idx+4]
    if version == 1:
        width_raw = struct.unpack('>I', data[offset+100:offset+104])[0]
        height_raw = struct.unpack('>I', data[offset+104:offset+108])[0]
    else:
        width_raw = struct.unpack('>I', data[offset+84:offset+88])[0]
        height_raw = struct.unpack('>I', data[offset+88:offset+92])[0]
    width = width_raw / 65536.0
    height = height_raw / 65536.0
    print(f'Resolution: {int(width)}x{int(height)}')
    print(f'File size: {len(data)/1024/1024:.1f} MB')

# Also check a few other recent videos for comparison
import glob, os
videos = sorted(glob.glob('/home/ubuntu/kg/data/files/videos/video_*.mp4'), key=os.path.getmtime, reverse=True)[:5]
for v in videos:
    with open(v, 'rb') as f:
        d = f.read()
    i = d.find(b'tkhd')
    if i >= 0:
        o = i - 4
        ver = d[i+4]
        if ver == 1:
            wr = struct.unpack('>I', d[o+100:o+104])[0]
            hr = struct.unpack('>I', d[o+104:o+108])[0]
        else:
            wr = struct.unpack('>I', d[o+84:o+88])[0]
            hr = struct.unpack('>I', d[o+88:o+92])[0]
        name = os.path.basename(v)
        print(f'{name}: {int(wr/65536)}x{int(hr/65536)} ({len(d)/1024/1024:.1f}MB)')
