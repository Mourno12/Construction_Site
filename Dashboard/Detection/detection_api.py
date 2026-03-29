#!/usr/bin/env python3
import sys
import json
import traceback

from image_detection import process_image
from video_detection import process_video
from webcam_detection import process_webcam


def main():
    try:
        if len(sys.argv) < 2:
            raise Exception("Mode required: image | video | webcam")

        mode = sys.argv[1]

        if mode == "image":
            input_path = sys.argv[2]
            output_path = sys.argv[3]
            result = process_image(input_path, output_path)

        elif mode == "video":
            input_path = sys.argv[2]
            output_path = sys.argv[3]
            result = process_video(input_path, output_path)

        elif mode == "webcam":
            result = process_webcam()

        else:
            raise Exception(f"Unknown mode: {mode}")

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()