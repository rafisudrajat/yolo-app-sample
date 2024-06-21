from ultralytics import YOLO
import argparse


# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some files.")

    # Add arguments
    parser.add_argument(
        'mode',
        type=str,
        choices=[
            'img_inference',
            'video_inference'],
        help='Mode operation to perform')
    parser.add_argument(
        'filename',
        type=str,
        help='The name of the file to process')
    parser.add_argument(
        '--save_image',
        action='store_true',
        help='Save image on image inference')
    parser.add_argument(
        '--output_image',
        type=str,
        help='Path to output image')

    # Parse the arguments
    args = parser.parse_args()
    if args.mode == "img_inference":
        if args.output_image == "" or args.output_image is None:
            image_inference(args.filename, args.save_image)
        else:
            image_inference(args.filename, args.save_image, args.output_image)
    elif args.mode == "video_inference":
        model.track(
            args.filename,
            show=True,
            tracker="bytetrack.yaml")


def image_inference(img_path: str, save_image: bool = False,
                    img_output_path: str = "artifact/output/img_output.jpg"):
    # Run inference on the source
    results = model(img_path)  # list of Results objects
    # Visualize the results
    for _, r in enumerate(results):
        # Show results to screen (in supported environments)
        r.show()

        # Save results to disk
        if save_image:
            r.save(filename=img_output_path)


if __name__ == "__main__":
    main()
