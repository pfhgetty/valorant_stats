import argparse

import numpy as np
import cv2

import detectors
import loader

from agent import PlayerInfo, AgentPictureInfo

# CLI Arguments
parser = argparse.ArgumentParser(description="Process a Valorant image")
parser.add_argument("--icons", help="Path for agent icons.", type=str, required=True)
parser.add_argument(
    "--image_path", help="Path for image to process.", type=str, required=True
)
args = parser.parse_args()


# TOP HUD CONSTANTS
# Detection Confidence Threshold for Agents
AGENT_CONF_THRESHOLD = 0.7
FIRST_BLUE_AGENT_POS = np.array([710, 30])
AGENT_SPACING = np.array([66, 0])
HEALTH_BAR_Y = 76
HEALTH_BAR_WIDTH_HEIGHT = np.array([40, 6])
# W x H
AGENT_IMAGE_SIZE = np.array([40, 40])
TOP_HUD_HEIGHT = 150
# Health bar can be white or red (in BGR format)
HEALTH_BAR_COLORS = np.array([[253, 253, 254], [91, 96, 241]], dtype=np.uint8)

# SCOREBOARD CONSTANTS
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


def show_debug_image(
    im,
    top_hud_and_scoreboard_pictures: list[tuple[AgentPictureInfo, AgentPictureInfo]],
    player_infos: list[PlayerInfo],
):
    health_offset = np.array([0, 65])
    scoreboard_y_text_offset = np.array([0, 10])
    scoreboard_text_offset = np.array([125, 0])
    colors_for_team = {"red": (0, 0, 255), "blue": (255, 0, 0)}
    for (top_hud_picture, scoreboard_picture), player_info in zip(
        top_hud_and_scoreboard_pictures, player_infos
    ):
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
    cv2.imshow("debug", im)
    cv2.waitKey(0)
    return


def top_hud_detectors(
    top_hud, blue_icons, red_icons
) -> tuple[list[AgentPictureInfo], list[int]]:
    _, w, _ = top_hud.shape
    FIRST_RED_AGENT_POS = (
        w - FIRST_BLUE_AGENT_POS[0] - AGENT_IMAGE_SIZE[0] + 1,
        FIRST_BLUE_AGENT_POS[1],
    )

    image_agent_infos = detectors.detect_all_agents(
        top_hud,
        blue_icons,
        red_icons,
        FIRST_BLUE_AGENT_POS,
        FIRST_RED_AGENT_POS,
        -AGENT_SPACING,
        AGENT_SPACING,
        AGENT_IMAGE_SIZE,
        AGENT_CONF_THRESHOLD,
    )
    agent_healths = detectors.detect_health_bar(
        top_hud,
        image_agent_infos,
        HEALTH_BAR_Y,
        HEALTH_BAR_WIDTH_HEIGHT,
        HEALTH_BAR_COLORS,
        DEBUG=False,
    )
    return image_agent_infos, agent_healths


def scoreboard_detectors(im, blue_icons):
    agent_pictures = detectors.detect_all_agents(
        im,
        blue_icons,
        blue_icons,
        SCOREBOARD_FIRST_BLUE_AGENT_POS,
        SCOREBOARD_FIRST_RED_AGENT_POS,
        SCOREBOARD_AGENT_SPACING,
        SCOREBOARD_AGENT_SPACING,
        SCOREBOARD_AGENT_IMAGE_SIZE,
        AGENT_CONF_THRESHOLD,
        # DEBUG=True
    )
    return agent_pictures, detectors.detect_scoreboard(
        im,
        agent_pictures,
        SCOREBOARD_ENTRY_HEIGHT,
        USERNAME_X_BOUNDS,
        ULTIMATE_X_BOUNDS,
        KILLS_X_BOUNDS,
        DEATHS_X_BOUNDS,
        ASSISTS_X_BOUNDS,
        CREDITS_X_BOUNDS,
    )

import time

def main(image_path, icon_directory):
    im = cv2.imread(image_path)

    # cv2.imshow("image", im)
    # cv2.waitKey(0)
    blue_icons, red_icons = loader.load_blue_red_icons(icon_directory, AGENT_IMAGE_SIZE)
    scoreboard_icons = loader.load_icons(icon_directory, SCOREBOARD_AGENT_IMAGE_SIZE)
    top_hud = im[:TOP_HUD_HEIGHT, :, :]

    
    top_hud_agent_pictures, agent_healths = top_hud_detectors(
        top_hud, blue_icons, red_icons
    )
    start_time = time.time()
    scoreboard_agent_pictures, (
        usernames,
        ultimate_statuses,
        kills,
        deaths,
        assists,
        credits,
    ) = scoreboard_detectors(im, scoreboard_icons)

    matched_pictures = dict()
    # Match agents from top hud to scoreboard
    for i, agent_picture_scoreboard in enumerate(scoreboard_agent_pictures):
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
            matched_pictures[(None, agent_picture_scoreboard)] = (
                usernames[i],
                ultimate_statuses[i],
                kills[i],
                deaths[i],
                assists[i],
                credits[i],
                0, # health is unknown so they're probably dead
            )
            continue
        match = top_hud_agent_pictures.pop(match_index)
        matched_pictures[(match, agent_picture_scoreboard)] = (
            usernames[i],
            ultimate_statuses[i],
            kills[i],
            deaths[i],
            assists[i],
            credits[i],
            agent_healths.pop(match_index),
        )

    agent_infos = []
    for (
        (agent_picture_top_hud, scoreboard_agent_picture),
        (username, ultimate_status, kill, death, assist, credit, health),
    ) in matched_pictures.items():
        agent_infos.append(
            PlayerInfo(
                username,
                ultimate_status,
                kill,
                death,
                assist,
                credit,
                scoreboard_agent_picture.agent_name,
                scoreboard_agent_picture.team,
                health,
            )
        )
    end_time = time.time()
    print(end_time - start_time)
    print(agent_infos)
    show_debug_image(im, matched_pictures.keys(), agent_infos)


if __name__ == "__main__":
    image_path = args.image_path
    icon_directory = args.icons
    main(image_path=image_path, icon_directory=icon_directory)
