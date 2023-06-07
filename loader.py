import os
import numpy as np
import cv2
from percep_hash import pHash

def load_left_right_icons(icon_directory, agent_image_size):
    left_icons = load_icons(icon_directory, agent_image_size)
    right_icons = flip_icons(left_icons)
    return hash_icons(left_icons), hash_icons(right_icons)


def load_icons(icon_directory, agent_image_size):
    icons = dict()
    icon_paths = []
    if os.path.isfile(icon_directory):
        icon_paths.append((os.path.basename(icon_directory), icon_directory))
    else:
        icon_paths = list(map(lambda x: (x, os.path.join(icon_directory, x)), os.listdir(icon_directory)))
    for filename, path in icon_paths:
        agent_name = filename.split("_")[0]
        icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        icon_image = icon[:, :, :3]
        icon_mask = icon[:, :, -1]
        if agent_image_size is not None:
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
            icon_image = icon_resized_masked
            icon_mask  = icon_mask_resized
        icons[agent_name] = (icon_image, icon_mask)
    return icons

def hash_icons(icons):
    hashed_icons = dict()
    for agent_name, (icon, icon_mask) in icons.items():
        hashed = pHash(icon)
        hashed_icons[agent_name] = (hashed, icon, icon_mask)
    return hashed_icons

def flip_icons(icons):
    flipped_icons = dict()
    for agent_name, (icon, icon_mask) in icons.items():
        icon_flipped = cv2.flip(icon, 1)
        icon_mask_flipped = cv2.flip(icon_mask, 1)
        flipped_icons[agent_name] = (icon_flipped, icon_mask_flipped)
    return flipped_icons