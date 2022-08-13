import argparse

import numpy as np
import cv2

import detectors
import loader

from agent import AgentInfo, ImageAgentInfo

# CLI Arguments
parser = argparse.ArgumentParser(description="Process a Valorant image")
parser.add_argument("--icons", help="Path for agent icons.", type=str, required=True)
parser.add_argument("--image_path", help="Path for image to process.", type=str, required=True)
args = parser.parse_args()

# Detection Confidence Threshold for Agents
AGENT_CONF_THRESHOLD = 0.75

FIRST_BLUE_AGENT_POS = np.array([710, 30])
AGENT_SPACING = np.array([66, 0])

HEALTH_BAR_Y = 76
HEALTH_BAR_WIDTH_HEIGHT = np.array([40, 6])
# W x H
AGENT_IMAGE_SIZE = np.array([40, 40])

TOP_HUD_HEIGHT = 150

# Health bar can be white or red (in BGR format)
HEALTH_BAR_COLORS = np.array([[253, 253, 254], [91, 96, 241]], dtype=np.uint8)

DEBUG_TEXT_OPTIONS = {
    "fontFace": cv2.FONT_HERSHEY_PLAIN,
    "fontScale": 0.5,
    "thickness": 1,
    "lineType": cv2.LINE_AA,
}


def show_debug_image(
    im, image_agent_infos: list[ImageAgentInfo], agent_infos: list[AgentInfo]
):
    health_offset = np.array([0, 65])
    colors_for_team = {"red": (0, 0, 255), "blue": (255, 0, 0)}
    for image_info, agent_info in zip(image_agent_infos, agent_infos):
        color = colors_for_team[agent_info.team]
        im = cv2.rectangle(
            im,
            image_info.image_rect[0],
            image_info.image_rect[1],
            color,
        )
        im = cv2.putText(
            im,
            agent_info.name,
            image_info.image_rect[0],
            color=color,
            **DEBUG_TEXT_OPTIONS,
        )
        im = cv2.putText(
            im,
            f"health: {str(agent_info.health)}",
            image_info.image_rect[0] + health_offset,
            color=color,
            **DEBUG_TEXT_OPTIONS,
        )
    cv2.imshow("debug", im)
    cv2.waitKey(0)
    return


def main(image_path, icon_directory):
    im = cv2.imread(image_path)

    # cv2.imshow("image", im)
    # cv2.waitKey(0)
    blue_icons, red_icons = loader.load_blue_red_icons(icon_directory, AGENT_IMAGE_SIZE)
    top_hud = im[:TOP_HUD_HEIGHT, :, :]
    image_agent_infos = detectors.detect_all_agents(
        top_hud,
        blue_icons,
        red_icons,
        FIRST_BLUE_AGENT_POS,
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
    agent_infos = []
    for image_agent_info, health in zip(image_agent_infos, agent_healths):
        agent_infos.append(
            AgentInfo(image_agent_info.name, image_agent_info.team, health)
        )

    print(agent_infos)
    show_debug_image(im, image_agent_infos, agent_infos)


if __name__ == "__main__":
    image_path = args.image_path
    icon_directory = args.icons
    main(image_path=image_path, icon_directory=icon_directory)
