"""Camera-independent automatic neutral-pose calibration and action counting."""
from collections import deque
from time import monotonic
import numpy as np

def relative_angle(value, baseline):
    return (value-baseline+180.)%360.-180.

def frontal_angle(value):
    # The template's axes can represent frontal pose near 180 rather than 0.
    return (value+90.)%180.-90.

class AutomaticActions:
    def __init__(self, direction="positive"):
        self.direction=1 if direction=="positive" else -1
        self.samples=deque(maxlen=80)
        self.baseline=None
        self.anchor=None
        self.ear_ref=None
        self.blink_count=self.head_down_count=self.head_up_count=0
        self.closed_frames=self.blink_flash=0
        self.head_state="NORMAL"
        self.normal_since=None
        self.last_seen=None
        self.calibration_progress=0.
        self.calibration_hint="自动校准：等待自然正视、睁眼并保持稳定"

    def reset_counts(self):
        self.blink_count=self.head_down_count=self.head_up_count=0
        self.closed_frames=self.blink_flash=0

    def missing(self, now=None):
        now=monotonic() if now is None else now
        self.closed_frames=0
        self.normal_since=None
        if self.baseline is None:
            self.samples.clear(); self.calibration_progress=0.
        # Do not bridge an action across a long loss, or assume a new identity.
        if self.last_seen is not None and now-self.last_seen>1.:
            self.head_state="WAIT_NORMAL"
        return "UNKNOWN","UNKNOWN",None,None

    def update_measurements(self, pitch, yaw, roll, ear, down=10., up=10., now=None):
        now=monotonic() if now is None else now
        if not np.all(np.isfinite([pitch,yaw,roll,ear])):
            return self.missing(now)
        self.last_seen=now
        frontal=(abs(frontal_angle(yaw))<20 and abs(frontal_angle(roll))<20)
        if self.baseline is None:
            if not frontal or abs(frontal_angle(pitch))>25 or ear<.16:
                self.samples.clear(); self.calibration_progress=0.
                self.calibration_hint="自动校准：等待自然正视、睁眼"
                return "CALIBRATING","CALIBRATING",pitch,ear
            if self.samples and abs(relative_angle(pitch,self.samples[-1][1]))>3:
                self.samples.clear()
            self.samples.append((now,pitch,ear))
            p0=self.samples[0][1]
            offsets=[relative_angle(v[1],p0) for v in self.samples]
            if np.ptp(offsets)>4:
                self.samples.clear(); self.samples.append((now,pitch,ear))
            elapsed=now-self.samples[0][0]
            self.calibration_progress=min(.99,elapsed/1.2)
            self.calibration_hint="自动校准：保持自然正视 %.0f%%" % (self.calibration_progress*100)
            if elapsed>=1.2 and len(self.samples)>=12:
                p0=self.samples[0][1]
                self.baseline=p0+float(np.median([relative_angle(v[1],p0) for v in self.samples]))
                self.anchor=self.baseline
                self.ear_ref=float(np.median([v[2] for v in self.samples]))
                self.calibration_progress=1.
                self.samples.clear()
            return "CALIBRATING","CALIBRATING",pitch,ear

        # Large side poses pause counting, while preserving the neutral reference.
        if abs(frontal_angle(yaw))>40 or abs(frontal_angle(roll))>35:
            self.closed_frames=0;self.normal_since=None
            self.head_state="WAIT_NORMAL"
            return "UNKNOWN","UNKNOWN",pitch,ear
        delta=self.direction*relative_angle(pitch,self.baseline)
        back=min(down,up)*.45
        closed=ear<.68*self.ear_ref
        if closed: self.closed_frames+=1
        else:
            if 1<=self.closed_frames<=8:
                self.blink_count+=1;self.blink_flash=6
            self.closed_frames=0
        eyes="BLINK" if self.blink_flash>0 else ("CLOSED" if closed else "OPEN")
        self.blink_flash=max(0,self.blink_flash-1)
        if self.head_state=="NORMAL":
            if delta>down:
                self.head_state="HEAD_DOWN";self.head_down_count+=1
            elif delta<-up:
                self.head_state="HEAD_UP";self.head_up_count+=1
        elif abs(delta)<back:
            self.head_state="NORMAL"

        # Adapt only near the initial neutral anchor, never during a held action.
        if (self.head_state=="NORMAL" and abs(delta)<3 and frontal and not closed):
            if self.normal_since is None: self.normal_since=now
            if now-self.normal_since>=2.:
                step=np.clip(relative_angle(pitch,self.baseline)*.005,-.01,.01)
                candidate=self.baseline+step
                if abs(relative_angle(candidate,self.anchor))<=3:
                    self.baseline=candidate
                self.ear_ref=.995*self.ear_ref+.005*ear
        else:
            self.normal_since=None
        head="UNKNOWN" if self.head_state=="WAIT_NORMAL" else self.head_state
        return head,eyes,pitch,ear
