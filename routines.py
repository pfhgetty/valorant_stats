import os
import cv2
import itertools
from mss import mss
import time
import multiprocessing
from queue import Queue
import cv2
import helpers
import numpy as np
from getch import getch
import _thread

TOP_HUD_HEIGHT = 150


def read_from_image_path(image_path, icon_directory, symbol_directory, save_debug=False):
    im = cv2.imread(image_path)
    # cv2.imshow("image", im)
    # cv2.waitKey(0)
    tesseract = Queue()
    for t in [
        helpers.create_tesseract() for _ in range(max(multiprocessing.cpu_count(), 1))
    ]:
        tesseract.put(t)

    blue_icons, red_icons, scoreboard_icons, symbol_icons = helpers.load_icons(
        icon_directory, symbol_directory
    )
    top_hud = im[:TOP_HUD_HEIGHT, :, :]

    # profile routine
    top_hud_agent_pictures = helpers.get_top_hud_positions(
        top_hud, blue_icons, red_icons
    )
    scoreboard_agent_pictures = helpers.get_scoreboard_positions(im, scoreboard_icons)

    start_time = time.time()
    agent_healths = helpers.get_healths(top_hud, top_hud_agent_pictures)
    scoreboard_infos = helpers.scoreboard_detectors(
        im, tesseract, scoreboard_agent_pictures, symbol_icons["Spike"]
    )
    matched_pictures = helpers.match_scoreboard_and_top_hud_data(
        scoreboard_agent_pictures,
        scoreboard_infos,
        top_hud_agent_pictures,
        agent_healths,
    )

    end_time = time.time()
    print(f"time: {end_time - start_time} sec")
    print(matched_pictures.values())
    debug_im = helpers.get_debug_image(
        im, matched_pictures.keys(), matched_pictures.values()
    )
    cv2.imshow("im", debug_im)
    cv2.waitKey(0)
    if save_debug:
        cv2.imwrite(save_debug, debug_im)
    return matched_pictures.values()


def input_thread(stop_signal, stop_char):
    while True:
        c = getch().decode().lower()
        if c == stop_char:
            stop_signal.append(True)
            break


class GetPositionInfo:
    def __init__(self):
        return

    def run(self, im, state_vars):
        top_hud = im[:TOP_HUD_HEIGHT, :, :]
        top_hud_agent_pictures = helpers.get_top_hud_positions(
            top_hud, state_vars["blue_icons"], state_vars["red_icons"]
        )
        scoreboard_agent_pictures = helpers.get_scoreboard_positions(
            im, state_vars["scoreboard_icons"]
        )
        if len(top_hud_agent_pictures) > 0 and len(scoreboard_agent_pictures) > 0:
            state_vars["top_hud_agent_pictures"] = top_hud_agent_pictures
            state_vars["scoreboard_agent_pictures"] = scoreboard_agent_pictures
            return LiveUpdate()
        return GetPositionInfo()


class LiveUpdate:
    def __init__(
        self, blue_icons, red_icons, scoreboard_icons, symbol_icons, tesseract
    ):
        self.blue_icons = blue_icons
        self.red_icons = red_icons
        self.scoreboard_icons = scoreboard_icons
        self.symbol_icons = symbol_icons
        self.tesseract = tesseract

    def update_positions(
        self,
        im,
    ):
        top_hud = im[:TOP_HUD_HEIGHT, :, :]
        top_hud_agent_pictures = helpers.get_top_hud_positions(
            top_hud, self.blue_icons, self.red_icons
        )
        scoreboard_agent_pictures = helpers.get_scoreboard_positions(
            im, self.scoreboard_icons
        )
        self.top_hud_agent_pictures = top_hud_agent_pictures
        self.scoreboard_agent_pictures = scoreboard_agent_pictures
        return len(top_hud_agent_pictures), len(scoreboard_agent_pictures)

    def update(self, im):
        top_hud = im[:TOP_HUD_HEIGHT, :, :]
        agent_healths = helpers.get_healths(top_hud, self.top_hud_agent_pictures)

        scoreboard_infos = helpers.scoreboard_detectors(
            im,
            self.tesseract,
            self.scoreboard_agent_pictures,
            self.symbol_icons["Spike"],
        )
        self.agent_healths = agent_healths
        self.scoreboard_infos = scoreboard_infos
        return helpers.match_scoreboard_and_top_hud_data(
            self.scoreboard_agent_pictures,
            self.scoreboard_infos,
            self.top_hud_agent_pictures,
            self.agent_healths,
        )


def read_from_screen_capture(icon_directory, symbol_directory, monitor_num):
    print("Press the 'S' key to stop")
    #
    stop_signal = []
    _thread.start_new_thread(input_thread, (stop_signal, "s"))

    tesseract = Queue()
    for t in [
        helpers.create_tesseract()
        for _ in range(max(multiprocessing.cpu_count() // 4, 1))
    ]:
        tesseract.put(t)

    blue_icons, red_icons, scoreboard_icons, symbol_icons = helpers.load_icons(
        icon_directory, symbol_directory
    )
    updater = LiveUpdate(
        blue_icons, red_icons, scoreboard_icons, symbol_icons, tesseract
    )
    writer = cv2.VideoWriter(
        "debug.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (
            1920,
            1080,
        ),
    )
    with mss() as capture:
        monitor = capture.monitors[monitor_num]
        print("Capturing agents in top HUD...")
        l1 = l2 = 0
        while (l1 < 1 or l2 < 1) and not stop_signal:
            im = np.ascontiguousarray(capture.grab(monitor))[:, :, :3]
            l1, l2 = updater.update_positions(im)
            writer.write(im)
        if not stop_signal:
            print("Starting live update...")
            while not stop_signal:
                im = np.ascontiguousarray(capture.grab(monitor))[:, :, :3]
                matched_pictures = updater.update(im)
                debug_im = helpers.get_debug_image(
                    im,
                    matched_pictures.keys(),
                    matched_pictures.values(),
                )
                writer.write(debug_im)
    writer.release()
