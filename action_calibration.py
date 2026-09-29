"""Automatic local-pose adaptation and action counting."""
from collections import deque
from time import monotonic
import numpy as np


def relative_angle(value, baseline):
    return (value-baseline+180.)%360.-180.


def frontal_angle(value):
    return (value+90.)%180.-90.


class AutomaticActions:
    """Track blinks and pitch gestures without an explicit frontal calibration."""

    def __init__(self, direction="positive", side_factor_scale=1.0):
        self.direction=1 if direction=="positive" else -1
        self.side_factor_scale=side_factor_scale
        self.samples=deque(maxlen=30)
        self.pitch_deltas=deque(maxlen=3)
        self.baseline=None
        self.anchor=None
        self.yaw_anchor=None
        self.ear_ref=None
        self.blink_count=self.head_down_count=self.head_up_count=0
        self.closed_frames=self.blink_flash=0
        self.eye_state="OPEN"
        self.closed_since=None
        self.reopen_since=None
        self.last_eye_pitch=None
        self.last_eye_time=None
        self.head_state="NORMAL"
        self.head_cycle="READY"
        self.head_pending_since=None
        self.last_confirmed_live=None
        self.head_return_since=None
        self.last_head_count_at=None
        self.last_head_kind=None
        self.head_opposite_armed=False
        self.normal_since=None
        self.last_seen=None
        self.last_pose=None
        self.last_pose_time=None
        self.reanchor_required=False
        self.calibration_progress=1.
        self.calibration_hint="姿态参考自动适应中"
        self.reset_settle_active=False
        self.reset_settle_since=None

    def begin_reference_reset(self):
        """Collect stable live frames before arming actions after a C-key reset."""
        self.baseline=None
        self.samples.clear()
        self.reset_settle_active=True
        self.reset_settle_since=None
        self.calibration_progress=0.
        self.head_state="NORMAL";self.head_cycle="WAIT_NORMAL"
        self.head_return_since=None
        self._reset_eye_event()

    def reset_counts(self):
        self.blink_count=self.head_down_count=self.head_up_count=0
        self.closed_frames=self.blink_flash=0
        self.eye_state="OPEN"
        self.closed_since=self.reopen_since=None
        self.last_eye_pitch=self.last_eye_time=None

    def _reset_eye_event(self):
        self.closed_frames=0
        self.eye_state="OPEN"
        self.closed_since=self.reopen_since=None
        self.last_eye_pitch=self.last_eye_time=None

    def missing(self, now=None):
        now=monotonic() if now is None else now
        self._reset_eye_event()
        self.normal_since=None
        self.samples.clear()
        self.pitch_deltas.clear()
        self.last_pose=self.last_pose_time=None
        if self.reset_settle_active:self.reset_settle_since=None
        # Short liveness/face dropouts must not rearm a held head gesture.
        if self.last_seen is None or now-self.last_seen>1.0:
            self.reanchor_required=True
            self.head_state="WAIT_NORMAL"
            self.head_return_since=None
        return "UNKNOWN","UNKNOWN",None,None

    def _initialize_local_reference(self,pitch,yaw,ear,now):
        self.baseline=self.anchor=pitch
        self.yaw_anchor=yaw
        self.ear_ref=max(float(ear),.10)
        self.last_pose=(pitch,yaw,0.)
        self.last_pose_time=now
        self.reanchor_required=False
        self.samples.clear()
        self.pitch_deltas.clear();self.pitch_deltas.extend((0.,0.,0.))
        self.head_state="NORMAL"
        self.head_cycle="READY"
        self.head_pending_since=None
        self.head_opposite_armed=False
        self.calibration_progress=1.

    def _update_blink(self,pitch,ear,now):
        moving=False
        if self.last_eye_pitch is not None and self.last_eye_time is not None:
            dt=now-self.last_eye_time
            if 0 < dt <= .5:
                pitch_step=abs(relative_angle(pitch,self.last_eye_pitch))
                moving=pitch_step>4.0 and pitch_step/dt>60.0
        self.last_eye_pitch,self.last_eye_time=pitch,now
        ratio=ear/max(self.ear_ref,1e-6)
        close_threshold,open_threshold=.68,.76
        min_closed,max_closed,reopen_confirm=.035,.65,.045
        if moving and (self.eye_state!="OPEN" or ratio<=close_threshold):
            self.eye_state="BLOCKED"
            self.closed_since=self.reopen_since=None
        if self.eye_state=="OPEN":
            if ratio<=close_threshold and not moving:
                self.eye_state="CLOSED";self.closed_since=now;self.reopen_since=None
        elif self.eye_state=="CLOSED":
            if self.closed_since is not None and now-self.closed_since>max_closed:
                self.eye_state="BLOCKED";self.closed_since=self.reopen_since=None
            elif ratio>=open_threshold:
                self.eye_state="REOPEN";self.reopen_since=now
        elif self.eye_state=="REOPEN":
            if ratio<=close_threshold:
                self.eye_state="CLOSED";self.reopen_since=None
            elif self.reopen_since is not None and now-self.reopen_since>=reopen_confirm:
                duration=self.reopen_since-(self.closed_since or self.reopen_since)
                if min_closed<=duration<=max_closed:
                    self.blink_count+=1;self.blink_flash=6
                self.eye_state="OPEN";self.closed_since=self.reopen_since=None
        elif self.eye_state=="BLOCKED":
            if ratio>=open_threshold:
                if self.reopen_since is None:self.reopen_since=now
                elif now-self.reopen_since>=reopen_confirm:
                    self.eye_state="OPEN";self.reopen_since=None
            else:self.reopen_since=None
        closed=self.eye_state!="OPEN"
        self.closed_frames=self.closed_frames+1 if closed else 0
        eyes="BLINK" if self.blink_flash>0 else ("CLOSED" if closed else "OPEN")
        self.blink_flash=max(0,self.blink_flash-1)
        return closed,eyes

    def _adapt_side_pose(self,pitch,yaw,ear,now,pitch_step,yaw_step):
        """Recenter silently after a turn, without treating yaw as a nod."""
        stable=pitch_step<2.5 and yaw_step<2.5
        if not stable:
            self.samples.clear()
            return False
        self.samples.append((now,pitch,yaw,ear))
        if now-self.samples[0][0]<.30 or len(self.samples)<3:
            return False
        p0=self.samples[0][1]
        self.baseline=p0+float(np.median([relative_angle(v[1],p0) for v in self.samples]))
        self.anchor=self.baseline
        self.yaw_anchor=float(np.median([v[2] for v in self.samples]))
        good_ears=[v[3] for v in self.samples if v[3]>.70*self.ear_ref]
        if good_ears:self.ear_ref=float(np.median(good_ears))
        self.samples.clear()
        self.pitch_deltas.clear();self.pitch_deltas.extend((0.,0.,0.))
        self.reanchor_required=False
        self.head_state="NORMAL"
        self.head_cycle="READY"
        self.head_pending_since=None
        return True

    def update_measurements(self,pitch,yaw,roll,ear,down=3.5,up=3.5,now=None,translation=False,count_enabled=True,confirmed_live=None):
        now=monotonic() if now is None else now
        if not np.all(np.isfinite([pitch,yaw,roll,ear])):
            return self.missing(now)
        self.last_seen=now
        recent_live=(self.last_confirmed_live is not None and
                     0<=now-self.last_confirmed_live<=.6)
        if confirmed_live is None:confirmed_live=count_enabled
        if confirmed_live:self.last_confirmed_live=now
        yaw_view=frontal_angle(yaw)
        roll_view=frontal_angle(roll)
        if self.reset_settle_active:
            if not count_enabled:
                self.samples.clear();self.reset_settle_since=None
                return "UNKNOWN","UNKNOWN",pitch,ear
            if self.samples:
                previous=self.samples[-1]
                if (abs(relative_angle(pitch,previous[1]))>2.5 or
                        abs(relative_angle(yaw,previous[2]))>3.5):
                    self.samples.clear();self.reset_settle_since=None
            if self.reset_settle_since is None:self.reset_settle_since=now
            self.samples.append((now,pitch,yaw,ear))
            self.calibration_progress=min(.99,(now-self.reset_settle_since)/.55)
            if now-self.reset_settle_since>=.55 and len(self.samples)>=4:
                p0=self.samples[0][1]
                pitch_ref=p0+float(np.median([relative_angle(v[1],p0) for v in self.samples]))
                yaw_ref=float(np.median([v[2] for v in self.samples]))
                ear_ref=float(np.median([v[3] for v in self.samples]))
                self._initialize_local_reference(pitch_ref,yaw_ref,ear_ref,now)
                self.reset_settle_active=False
            return "NORMAL","OPEN",pitch,ear
        if self.baseline is None:
            self._initialize_local_reference(pitch,yaw,ear,now)
            return "NORMAL","OPEN",pitch,ear

        if count_enabled:
            closed,eyes=self._update_blink(pitch,ear,now)
        else:
            self._reset_eye_event()
            closed,eyes=False,"UNKNOWN"
        if abs(yaw_view)>70 or abs(roll_view)>55:
            self.samples.clear();self.reanchor_required=True
            self.head_state="NORMAL";self.head_cycle="WAIT_NORMAL"
            self.last_pose=(pitch,yaw,roll);self.last_pose_time=now
            return "UNKNOWN",eyes,pitch,ear

        pitch_step=yaw_step=0.
        dt=None
        if self.last_pose is not None and self.last_pose_time is not None:
            dt=now-self.last_pose_time
            pitch_step=abs(relative_angle(pitch,self.last_pose[0]))
            yaw_step=abs(relative_angle(yaw,self.last_pose[1]))
        self.last_pose=(pitch,yaw,roll);self.last_pose_time=now

        yaw_offset=abs(relative_angle(yaw,self.yaw_anchor))
        turning=(dt is not None and 0<dt<=.5 and yaw_step>7.0 and yaw_step/dt>65.)
        if translation:
            self.samples.clear();self.pitch_deltas.clear()
            self.reanchor_required=True
            self.head_state="NORMAL";self.head_cycle="WAIT_NORMAL"
            return "NORMAL",eyes,pitch,ear
        if turning or yaw_offset>20.:
            self.reanchor_required=True
            self.head_state="NORMAL";self.head_cycle="WAIT_NORMAL"
        if self.reanchor_required:
            self._adapt_side_pose(pitch,yaw,ear,now,pitch_step,yaw_step)
            return "NORMAL",eyes,pitch,ear

        raw_delta=self.direction*relative_angle(pitch,self.baseline)
        self.pitch_deltas.append(raw_delta)
        # Side views need a larger threshold because pitch/yaw coupling grows.
        side=abs(yaw_view)
        if side<20.: pose_factor=1.0
        elif side<40.: pose_factor=1.35
        elif side<55.: pose_factor=1.70
        else: pose_factor=2.05
        pose_factor=1.0+(pose_factor-1.0)*self.side_factor_scale
        down_trigger=down*pose_factor
        up_trigger=up*pose_factor
        # Trigger on the current frame so a quick nod is not lost at low FPS.
        delta=raw_delta
        release=min(down_trigger,up_trigger)*.45
        if self.head_cycle=="WAIT_NORMAL":
            self.head_state="NORMAL"
            if abs(delta)<release:
                if self.head_return_since is None:self.head_return_since=now
                elif now-self.head_return_since>=.40:
                    self.head_cycle="READY";self.head_return_since=None
            else:self.head_return_since=None
        elif self.head_cycle=="READY":
            kind="DOWN" if delta>down_trigger else ("UP" if delta<-up_trigger else None)
            if kind is not None:
                self.head_cycle=kind if count_enabled else "PENDING_"+kind
                self.head_state="HEAD_"+kind if count_enabled else "NORMAL"
                self.head_pending_since=now if not count_enabled else None
                self.head_return_since=None
                self.head_opposite_armed=False
                if count_enabled:
                    if kind=="DOWN":self.head_down_count+=1
                    else:self.head_up_count+=1
                    self.last_head_count_at=now
                    self.last_head_kind=kind
        elif self.head_cycle.startswith("PENDING_"):
            kind=self.head_cycle.removeprefix("PENDING_")
            still_active=(delta>down_trigger if kind=="DOWN" else delta<-up_trigger)
            if (count_enabled and recent_live and still_active and
                    now-self.head_pending_since<=.6):
                self.head_cycle=kind;self.head_state="HEAD_"+kind
                if kind=="DOWN":self.head_down_count+=1
                else:self.head_up_count+=1
                self.last_head_count_at=now;self.last_head_kind=kind;self.head_pending_since=None
            elif not still_active or now-self.head_pending_since>.6:
                self.head_cycle="RETURNING";self.head_state="NORMAL"
                self.head_return_since=now if abs(delta)<release else None
        elif self.head_cycle in ("DOWN","UP","RETURNING"):
            # The visible state returns to normal immediately.  The cycle stays
            # locked until neutral is sustained, so rebound cannot count as UP.
            if abs(delta)<release or (self.head_cycle=="DOWN" and delta<0) or (self.head_cycle=="UP" and delta>0):
                self.head_cycle="RETURNING";self.head_state="NORMAL"
                if abs(delta)<release:
                    if self.head_return_since is None:self.head_return_since=now
                    elif now-self.head_return_since>=.40 and now-self.last_head_count_at>=.80:
                        self.head_cycle="READY";self.head_return_since=None
                    elif now-self.head_return_since>=.18:
                        self.head_opposite_armed=True
                else:
                    self.head_return_since=None
                    self.head_opposite_armed=False
            elif self.head_cycle=="RETURNING":
                # A real opposite gesture can follow a short neutral pause.
                # Rebound immediately after the previous count remains locked.
                opposite=("UP" if self.last_head_kind=="DOWN" else "DOWN")
                opposite_active=(delta<-up_trigger if opposite=="UP" else delta>down_trigger)
                neutral_seen=self.head_opposite_armed
                cooled=(self.last_head_count_at is not None and
                        now-self.last_head_count_at>=.55)
                if opposite_active and neutral_seen and cooled and count_enabled:
                    self.head_cycle=opposite;self.head_state="HEAD_"+opposite
                    if opposite=="DOWN":self.head_down_count+=1
                    else:self.head_up_count+=1
                    self.last_head_count_at=now;self.last_head_kind=opposite
                    self.head_return_since=None
                    self.head_opposite_armed=False
                else:
                    self.head_state="NORMAL"
                    self.head_return_since=None
                    self.head_opposite_armed=False
            else:
                self.head_state="HEAD_"+self.head_cycle
                self.head_return_since=None

        # Slowly follow small posture drift only while no gesture is active.
        if self.head_cycle=="READY" and abs(delta)<release and not closed:
            alpha=min(.015,max(0.,(dt or 0.))*0.05)
            self.baseline+=np.clip(relative_angle(pitch,self.baseline)*alpha,-.03,.03)
            if ear>.76*self.ear_ref:self.ear_ref=.997*self.ear_ref+.003*ear
        head=self.head_state
        return head,eyes,pitch,ear
