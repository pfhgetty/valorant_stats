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

args = parser.parse_args()


if __name__ == "__main__":
    symbols_directory = args.symbols
    icon_directory = args.icons
    monitor = args.monitor
    routines.read_from_screen_capture(
        icon_directory=icon_directory,
        symbol_directory=symbols_directory,
        monitor_num=monitor,
    )

    # routines.read_from_image_path("./sample_images/test6.png", icon_directory, symbols_directory)
