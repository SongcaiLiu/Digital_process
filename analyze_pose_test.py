"""Regenerate a diagnostic report without opening the camera."""
import argparse
from diagnostic_recording import summarize
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder")
    args=p.parse_args()
    print(summarize(args.folder).resolve())
