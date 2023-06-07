from re import A
from turtle import position
import numpy as np
import cv2
from agent import AgentPictureInfo, PlayerInfo, PlayerInfoAndImages
import detectors
import loader

from tesserocr import PyTessBaseAPI, PSM, OEM

# TOP HUD CONSTANTS
# Detection Confidence Threshold for Agents
AGENT_CONF_THRESHOLD = 0.5
AGENT_SQDIFF_THRESHOLD = 0.15
FIRST_LEFT_AGENT_POS = np.array([710, 30])
AGENT_SPACING = np.array([66, 0])
HEALTH_BAR_Y = 76
HEALTH_BAR_WIDTH_HEIGHT = np.array([40, 6])
# W x H
AGENT_IMAGE_SIZE = np.array([40, 40])
# Health bar can be white or red (in BGR format)
HEALTH_BAR_COLORS = np.array([[250, 250, 250], [91, 96, 241]], dtype=np.uint8)

# SCOREBOARD CONSTANTS
SCOREBOARD_X_BOUNDS = np.array([570, 572+38])
SCOREBOARD_Y_BOUNDS = np.array([186, 875])
SCOREBOARD_AGENT_IMAGE_SIZE = np.array([33, 33])
SCOREBOARD_AGENT_SPACING = np.array([0, 34])
SCOREBOARD_FIRST_BLUE_AGENT_POS = np.array([573, 407])
SCOREBOARD_FIRST_RED_AGENT_POS = np.array([571, 638])
SCOREBOARD_ENTRY_HEIGHT = 33
USERNAME_X_BOUNDS = np.array([643, 808])
ULTIMATE_X_BOUNDS = np.array([813, 932])
KILLS_X_BOUNDS = np.array([934, 973])
DEATHS_X_BOUNDS = np.array([973, 1012])
ASSISTS_X_BOUNDS = np.array([1012, 1051])
CREDITS_X_BOUNDS = np.array([1205, 1287])

SYMBOL_IMAGE_SIZE = np.array([40, 34])
SYMBOL_CONF_THRESHOLD = 0.8


DEBUG_TEXT_OPTIONS = {
    "fontFace": cv2.FONT_HERSHEY_PLAIN,
    "fontScale": 0.8,
    "thickness": 1,
    "lineType": cv2.LINE_AA,
}
DEBUG_TEXT_OUTLINE_OPTIONS = {
    "fontFace": cv2.FONT_HERSHEY_PLAIN,
    "fontScale": 0.8,
    "thickness": 2,
    "color": (0, 0, 0),
    "lineType": cv2.LINE_AA,
}


def outlined_text(im, text, position, color):
    im = cv2.putText(im, text, position, **DEBUG_TEXT_OUTLINE_OPTIONS)
    im = cv2.putText(im, text, position, color=color, **DEBUG_TEXT_OPTIONS)
    return im


def get_debug_image(
    im,
    player_info_and_images: list[PlayerInfoAndImages]
):
    im = np.ascontiguousarray(im)
    health_offset = np.array([0, 65])
    scoreboard_y_text_offset = np.array([0, 10])
    scoreboard_text_offset = np.array([125, 0])
    colors_for_team = {"red": (0, 0, 255), "blue": (255, 0, 0)}
    for info in player_info_and_images:
        top_hud_picture = info.top_hud_picture
        scoreboard_picture = info.scoreboard_picture
        player_info = info.player
        color = colors_for_team[player_info.team]
        if top_hud_picture is not None:
            # TOP HUD INFORMATION
            im = cv2.rectangle(
                im,
                top_hud_picture.image_rect[0],
                top_hud_picture.image_rect[1],
                color,
            )
            im = outlined_text(
                im,
                player_info.agent_name,
                top_hud_picture.image_rect[0],
                color,
            )
            im = outlined_text(
                im,
                f"hp: {str(player_info.health)}",
                top_hud_picture.image_rect[0] + health_offset,
                color=color,
            )
        # SCOREBOARD INFORMATION
        im = cv2.rectangle(
            im,
            scoreboard_picture.image_rect[0],
            scoreboard_picture.image_rect[1],
            color,
        )

        im = outlined_text(
            im,
            player_info.username,
            scoreboard_picture.image_rect[0] + scoreboard_y_text_offset,
            color=color,
        )
        im = outlined_text(
            im,
            scoreboard_picture.agent_name,
            scoreboard_picture.image_rect[0] + scoreboard_y_text_offset * 2,
            color=color,
        )
        im = outlined_text(
            im,
            f"Spike: {player_info.spike_status}",
            scoreboard_picture.image_rect[0] + scoreboard_y_text_offset * 3,
            color=color,
        )
        im = outlined_text(
            im,
            f"ultimate: {str(player_info.ultimate_status)}",
            scoreboard_picture.image_rect[0]
            + scoreboard_y_text_offset
            + scoreboard_text_offset * 2,
            color=color,
        )
        im = outlined_text(
            im,
            f"kda: {str(player_info.kills)}/{str(player_info.deaths)}/{str(player_info.assists)}",
            scoreboard_picture.image_rect[0]
            + scoreboard_y_text_offset
            + scoreboard_text_offset * 3,
            color=color,
        )
        im = outlined_text(
            im,
            f"credits: {str(player_info.credits)}",
            scoreboard_picture.image_rect[0]
            + scoreboard_y_text_offset
            + scoreboard_text_offset * 5,
            color=color,
        )
    return im


def load_icons(icon_directory, symbol_directory):
    # Use hashes for agent icons in order to speed up matching
    left_icons, right_icons = loader.load_left_right_icons(icon_directory, AGENT_IMAGE_SIZE)
    scoreboard_icons = loader.hash_icons(
        loader.load_icons(icon_directory, SCOREBOARD_AGENT_IMAGE_SIZE)
    )

    spike_icon = loader.load_icons(symbol_directory + "/Spike_icon.png", SYMBOL_IMAGE_SIZE)["Spike"]
    chicken_icon = loader.load_icons(symbol_directory + "/Chickens_text.png", None)["Chickens"]
    symbol_icons = {"Spike" : spike_icon, "Chickens": chicken_icon}
    return left_icons, right_icons, scoreboard_icons, symbol_icons


def get_top_hud_positions(top_hud, left_icons, right_icons):
    _, w, _ = top_hud.shape
    FIRST_RIGHT_AGENT_POS = (
        w - FIRST_LEFT_AGENT_POS[0] - AGENT_IMAGE_SIZE[0] + 1,
        FIRST_LEFT_AGENT_POS[1],
    )
    image_agent_infos = detectors.detect_all_agents(
        top_hud,
        left_icons,
        right_icons,
        FIRST_LEFT_AGENT_POS,
        FIRST_RIGHT_AGENT_POS,
        -AGENT_SPACING,
        AGENT_SPACING,
        AGENT_IMAGE_SIZE,
        AGENT_CONF_THRESHOLD,
    )
    return image_agent_infos


def get_healths(
    top_hud, agent_picture_infos
) -> tuple[list[AgentPictureInfo], list[int]]:
    agent_healths = detectors.detect_health_bar(
        top_hud,
        agent_picture_infos,
        HEALTH_BAR_Y,
        HEALTH_BAR_WIDTH_HEIGHT,
        HEALTH_BAR_COLORS,
        DEBUG=False,
    )
    return agent_healths


def get_scoreboard_positions(im, icons):
    scoreboard = im[
        SCOREBOARD_Y_BOUNDS[0] : SCOREBOARD_Y_BOUNDS[1],
        SCOREBOARD_X_BOUNDS[0] : SCOREBOARD_X_BOUNDS[1],
        :,
    ]
    agent_pictures = detectors.find_all_agents(
        scoreboard,
        icons,
        SCOREBOARD_AGENT_IMAGE_SIZE,
        AGENT_SQDIFF_THRESHOLD,
        SCOREBOARD_X_BOUNDS,
        SCOREBOARD_Y_BOUNDS,
    )
    return agent_pictures

def update_scoreboard_agent_pictures(im, icons, past_agent_pictures: list[AgentPictureInfo]):
    positions = []
    for agent_picture in past_agent_pictures:
        positions.append(agent_picture.image_rect)

    agent_pictures = detectors.detect_agent_at_positions_on_team(
        im,
        icons,
        positions,
        0.25,
        past_agent_pictures,
    )
    return agent_pictures


def get_scoreboard_positions_static(im, icons):
    agent_pictures = detectors.detect_all_agents(
        im,
        icons,
        icons,
        SCOREBOARD_FIRST_BLUE_AGENT_POS,
        SCOREBOARD_FIRST_RED_AGENT_POS,
        SCOREBOARD_AGENT_SPACING,
        SCOREBOARD_AGENT_SPACING,
        SCOREBOARD_AGENT_IMAGE_SIZE,
        AGENT_CONF_THRESHOLD,
        # DEBUG=True
    )
    return agent_pictures


def scoreboard_detectors(im, tesseract, agent_pictures, spike_icon, chickens):
    return detectors.detect_scoreboard(
        im,
        tesseract,
        agent_pictures,
        spike_icon,
        chickens,
        SYMBOL_CONF_THRESHOLD,
        SCOREBOARD_ENTRY_HEIGHT,
        USERNAME_X_BOUNDS,
        ULTIMATE_X_BOUNDS,
        KILLS_X_BOUNDS,
        DEATHS_X_BOUNDS,
        ASSISTS_X_BOUNDS,
        CREDITS_X_BOUNDS,
    )


def match_scoreboard_and_top_hud_data(
    scoreboard_agent_pictures, scoreboard_infos, top_hud_agent_pictures, agent_healths
):
    # Copy these lists because we are going to mutate them
    agent_healths = list(agent_healths)
    top_hud_agent_pictures = list(top_hud_agent_pictures)

    matched_pictures = dict()
    # Match agents from top hud to scoreboard
    for i, (agent_picture_scoreboard, board_row) in enumerate(
        zip(scoreboard_agent_pictures, scoreboard_infos)
    ):
        match_index = next(
            (
                i
                for i, x in enumerate(top_hud_agent_pictures)
                if (
                    x.agent_name == agent_picture_scoreboard.agent_name
                    and x.team == agent_picture_scoreboard.team
                )
            ),
            None,
        )
        # assert (
        #     match_index is not None
        # ), f"{agent_picture_scoreboard} did not have match in {top_hud_agent_pictures}"
        if match_index is None:
            matched_pictures[(None, agent_picture_scoreboard)] = PlayerInfo(
                board_row.username,
                board_row.ultimate_status,
                board_row.kills,
                board_row.deaths,
                board_row.assists,
                board_row.credits,
                board_row.spike_status,
                agent_picture_scoreboard.agent_name,
                agent_picture_scoreboard.team,
                0,  # No match so they're probably dead
            )
            continue
        match = top_hud_agent_pictures.pop(match_index)
        matched_pictures[(match, agent_picture_scoreboard)] = PlayerInfo(
            board_row.username,
            board_row.ultimate_status,
            board_row.kills,
            board_row.deaths,
            board_row.assists,
            board_row.credits,
            board_row.spike_status,
            agent_picture_scoreboard.agent_name,
            agent_picture_scoreboard.team,
            agent_healths.pop(match_index),
        )
    return matched_pictures_formatted(matched_pictures)


def create_tesseract():
    tesseract = PyTessBaseAPI(
        lang='eng', path="./tessdata", psm=PSM.SINGLE_LINE, oem=OEM.DEFAULT
    )
    tesseract.SetVariable("tessedit_do_invert", "0")
    return tesseract

def matched_pictures_formatted(matched_pictures):
    out = []
    for (top_hud_picture, scoreboard_picture), player_info in matched_pictures.items():
        out.append(PlayerInfoAndImages(top_hud_picture, scoreboard_picture, player_info))
    return out
