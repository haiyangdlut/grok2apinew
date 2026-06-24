import struct, glob, os, time
videos = sorted(glob.glob('/home/ubuntu/kg/data/files/videos/video_*.mp4'), key=os.path.getmtime)
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
        mtime = os.path.getmtime(v)
        print(f'{os.path.basename(v)[:35]:40s} {int(wr/65536)}x{int(hr/65536)}  {time.strftime("%H:%M", time.localtime(mtime))}')
