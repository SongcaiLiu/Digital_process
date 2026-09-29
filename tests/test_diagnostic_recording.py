import csv
from tempfile import TemporaryDirectory
import numpy as np
from diagnostic_recording import DiagnosticRecorder
with TemporaryDirectory() as tmp:
    r=DiagnosticRecorder(tmp,dict(depth_scale=.001),2)
    rgb=np.zeros((24,32,3),np.uint8);depth=np.full((24,32),600,np.uint16)
    for i in range(8):
        if i==4:r.set_segment("7")
        r.record(rgb,depth,None,None,None,{},dict(elapsed_s=i*.25,classification="UNKNOWN",reason="中文原因"))
    r.close()
    assert r.error is None
    with (r.folder/"frames.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert rows[0]["expected"]=="LIVE" and rows[-1]["expected"]=="PHOTO"
    samples=[row for row in rows if row["sample"]]
    assert len(samples)==4
    with np.load(r.folder/samples[0]["sample"]) as saved:
        assert np.array_equal(saved["depth"],depth)
        assert np.array_equal(saved["rgb"],rgb)
    assert (r.folder/"report.md").read_text(encoding="utf-8").count(chr(10))>10
print("PASS: recorder labels, raw RGB-D, cadence and Chinese report")
