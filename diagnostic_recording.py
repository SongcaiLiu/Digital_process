"""Asynchronous RGB-D diagnostic recorder; never changes classifier thresholds."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
import cv2
import numpy as np

FIELDS=["frame","elapsed_s","camera_ms","segment","expected","classification","reason",
        "person_detected","face_detected","feature_status","pitch_raw","yaw_raw","roll_raw",
        "pitch_camera","yaw_camera","roll_camera","relative_pitch","head_state","eye_state",
        "calibrated","face_distance_m","valid_ratio","live_valid_ratio","spread_mm","support",
        "spatial_mm","noise_mm","nose_mm","curve_gain","cells","cheeks_available",
        "nose_available","missing_ratio","too_near_ratio","too_far_ratio","sample"]
SEGMENTS={
    "1":("正视","LIVE"), "2":("左右转头","LIVE"),
    "3":("低头抬头","LIVE"), "4":("组合姿态","LIVE"),
    "5":("照片","PHOTO"), "6":("备用真人段","LIVE"),
    "7":("打印照片","PHOTO"), "8":("屏幕图片或视频","PHOTO")}

def camera_angle(value):
    return (value+90.)%180.-90.

def summarize(folder):
    folder=Path(folder)
    groups=defaultdict(Counter); reasons=defaultdict(Counter)
    angle_groups=defaultdict(Counter)
    with (folder/"frames.csv").open(encoding="utf-8-sig",newline="") as handle:
        for row in csv.DictReader(handle):
            key=(row["segment"],row["expected"])
            groups[key]["总帧数"]+=1
            groups[key][row["classification"]]+=1
            reasons[key][row["reason"]]+=1
            if row["pitch_camera"] and row["yaw_camera"]:
                pitch=float(row["pitch_camera"]); yaw=float(row["yaw_camera"])
                angle=(row["expected"],int(np.floor(pitch/10))*10,int(np.floor(yaw/10))*10)
                angle_groups[angle][row["classification"]]+=1
    lines=["# 姿态测试报告","",
           "按人工标记统计，未知不算识别成功。角度为通用模板估计的相机角度，可能存在偏差。",
           "人脸丢失时没有角度，因此不会进入角度分桶；仍计入人工动作段。","",
           "|动作段|真实类型|帧数|真人|照片|未知|判对比例|",
           "|---|---|---:|---:|---:|---:|---:|"]
    for (segment,expected),count in groups.items():
        total=count["总帧数"]
        lines.append(f"|{segment}|{expected}|{total}|{count['LIVE']}|{count['PHOTO']}|{count['UNKNOWN']}|{count[expected]/total:.1%}|")
    lines.extend(["","## 各动作段主要原因",""])
    for (segment,expected),count in groups.items():
        lines.append(segment+"："+ "；".join(f"{reason}（{n}帧）" for reason,n in reasons[(segment,expected)].most_common(5)))
        lines.append("")
    lines.extend(["","## 俯仰/左右角度区间（每10度）","",
                  "|真实类型|俯仰区间|左右区间|真人|照片|未知|",
                  "|---|---|---|---:|---:|---:|"])
    for (expected,pitch,yaw),count in sorted(angle_groups.items()):
        lines.append(f"|{expected}|{pitch}～{pitch+10}|{yaw}～{yaw+10}|{count['LIVE']}|{count['PHOTO']}|{count['UNKNOWN']}|")
    target=folder/"report.md"
    target.write_text("\n".join(lines),encoding="utf-8")
    return target

class DiagnosticRecorder:
    def __init__(self,root,metadata,sample_fps=2.):
        if sample_fps<=0: raise ValueError("采样频率必须大于0")
        self.folder=Path(root)/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (self.folder/"samples").mkdir(parents=True)
        metadata.update(sample_fps=sample_fps,rgb_format="BGR uint8",
                        depth_format="aligned uint16; meters = raw * depth_scale",
                        landmark_format="RGB pixel coordinates; third coordinate is MediaPipe value")
        (self.folder/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        self.queue=Queue(maxsize=64)
        self.interval=1/sample_fps;self.last_sample=-1e9
        self.frame=0;self.dropped=0;self.error=None
        self.segment,self.expected=SEGMENTS["1"]
        self.worker=Thread(target=self._work,daemon=True);self.worker.start()
        print("测试记录目录："+str(self.folder.resolve()))

    def set_segment(self,key):
        if key in SEGMENTS:self.segment,self.expected=SEGMENTS[key]

    def record(self,color,depth,landmarks,person_box,face_box,geometry,values):
        self.frame+=1
        row={key:geometry.get(key,"") for key in FIELDS}
        row.update(values,frame=self.frame,segment=self.segment,expected=self.expected,sample="")
        sample=None
        if values["elapsed_s"]-self.last_sample>=self.interval:
            name=f"{self.frame:07d}.npz"
            row["sample"]="samples/"+name
            sample=(self.folder/row["sample"],color.copy(),depth.copy(),
                    np.empty((0,3)) if landmarks is None else landmarks.copy(),
                    np.asarray(person_box if person_box is not None else [],dtype=int),
                    np.asarray(face_box if face_box is not None else [],dtype=int))
        try:
            self.queue.put_nowait((row,sample))
            if sample:self.last_sample=values["elapsed_s"]
        except Exception as exc:
            from queue import Full
            if isinstance(exc,Full):self.dropped+=1
            else:raise

    def _work(self):
        try:
            with (self.folder/"frames.csv").open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=FIELDS,extrasaction="ignore")
                writer.writeheader()
                while True:
                    item=self.queue.get()
                    if item is None:break
                    row,sample=item
                    if sample:
                        path,rgb,depth,landmarks,person_box,face_box=sample
                        np.savez_compressed(path,rgb=rgb,depth=depth,landmarks=landmarks,
                                            person_box=person_box,face_box=face_box)
                    writer.writerow(row);handle.flush()
        except Exception as exc:self.error=str(exc)

    def close(self):
        # Avoid a deadlock if disk failure stopped the worker.
        from queue import Full
        while self.worker.is_alive():
            try:self.queue.put(None,timeout=.1);break
            except Full:continue
        self.worker.join()
        (self.folder/"recording_status.json").write_text(
            json.dumps(dict(observed_frames=self.frame,dropped_frames=self.dropped,error=self.error),indent=2),
            encoding="utf-8")
        if self.error:print("测试记录失败："+self.error)
        else:print("测试报告："+str(summarize(self.folder).resolve()))
        if self.dropped:print("记录队列溢出，丢弃帧数："+str(self.dropped))
