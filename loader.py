import os
import numpy as np
import cv2


def load_blue_red_icons(icon_directory, agent_image_size):
    blue_icons = load_icons(icon_directory, agent_image_size)
    red_icons = flip_icons(blue_icons)
    return blue_icons, red_icons


def load_icons(icon_directory, agent_image_size):
    icons = dict()
    for path in os.listdir(icon_directory):
        filename = os.path.join(icon_directory, path)
        # check if current filename is actually a file
        if os.path.isfile(filename):
            agent_name = path.split("_")[0]
            icon = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            icon_image = icon[:, :, :3]
            icon_mask = icon[:, :, -1]

            icon_resized = cv2.resize(
                icon_image, dsize=agent_image_size, interpolation=cv2.INTER_AREA
            )

            icon_mask_resized = cv2.resize(
                icon_mask, dsize=agent_image_size, interpolation=cv2.INTER_AREA
            )
            icon_mask_resized = (np.clip(icon_mask_resized, 254, 255) - 254) * 255

            icon_resized_masked = cv2.bitwise_and(
                icon_resized, icon_resized, mask=icon_mask_resized
            )
            icons[agent_name] = (icon_resized_masked, icon_mask_resized)

    return icons


def flip_icons(icons):
    flipped_icons = dict()
    for agent_name, (icon, icon_mask) in icons.items():
        icon_flipped = cv2.flip(icon, 1)
        icon_mask_flipped = cv2.flip(icon_mask, 1)
        flipped_icons[agent_name] = (icon_flipped, icon_mask_flipped)
    return flipped_icons
