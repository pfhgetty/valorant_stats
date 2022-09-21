import argparse
import routines

# CLI Arguments
parser = argparse.ArgumentParser(description="Process a Valorant image")
parser.add_argument("--icons", help="Path for agent icons.", type=str, required=True)
parser.add_argument(
    "--symbols", help="Path for symbolic icons.", type=str, required=True
)
parser.add_argument(
    "--monitor",
    help="Which monitor to screencap (monitor number starts at 1)",
    type=int,
    required=True,
)
parser.add_argument(
    "--output",
    help="Filepath to output data to.",
    type=str,
    required=True,
)

parser.add_argument(
    "--video_debug", help="Filepath to output debug video to.", type=str, required=False
)

args = parser.parse_args()


if __name__ == "__main__":
    symbols_directory = args.symbols
    icon_directory = args.icons
    monitor = args.monitor
    txt_file = args.output
    video_debug = args.video_debug
    routines.read_from_screen_capture(
        icon_directory=icon_directory,
        symbol_directory=symbols_directory,
        monitor_num=monitor,
        out_txt_file=txt_file,
        video_debug=video_debug,
    )

    # routines.read_from_image_path("./sample_images/test (3).jpg", icon_directory, symbols_directory, "demo/demo4.png")
