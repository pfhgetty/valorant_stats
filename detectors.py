import cv2
import numpy as np
from agent import ImageAgentInfo


def classify_agent(im, icons):
    agents = dict()
    for agent_name, (icon, mask) in icons.items():
        result = cv2.matchTemplate(im, icon, cv2.TM_CCOEFF_NORMED, mask=mask)
        agents[agent_name] = result.flatten()
    return agents


def detect_all_agents(
    top_hud,
    blue_icons,
    red_icons,
    first_blue_agent_pos,
    spacing,
    agent_image_size,
    agent_conf_threshold,
    DEBUG=False,
):
    _, w, _ = top_hud.shape
    agents = detect_agents_on_team(
        top_hud,
        blue_icons,
        first_blue_agent_pos,
        -spacing,
        agent_image_size,
        agent_conf_threshold,
        "blue",
        DEBUG=(255, 0, 0) if DEBUG else None,
    )
    first_red_agent_pos = (
        w - first_blue_agent_pos[0] - agent_image_size[0] + 1,
        first_blue_agent_pos[1],
    )
    agents += detect_agents_on_team(
        top_hud,
        red_icons,
        first_red_agent_pos,
        spacing,
        agent_image_size,
        agent_conf_threshold,
        "red",
        DEBUG=(0, 0, 255) if DEBUG else None,
    )
    return agents


def detect_agents_on_team(
    im,
    icons,
    first_icon_pos,
    spacing,
    agent_image_size,
    agent_conf_threshold,
    team,
    DEBUG=None,
):
    # Get initial positions to detect
    positions = []
    for i in range(5):
        bl = first_icon_pos + spacing * i
        tr = bl + agent_image_size
        positions.append((bl, tr))

    # Get initial detections
    agents = []
    for bl, tr in positions:
        im_part = im[bl[1] : tr[1], bl[0] : tr[0]]
        # if DEBUG is not None:
        #     cv2.imshow("part", im_part)
        #     cv2.waitKey(25)
        agent_confs = classify_agent(im_part, icons)

        agent_name, conf = max(agent_confs.items(), key=lambda x: x[1])
        if conf < agent_conf_threshold:
            continue
        agents.append(ImageAgentInfo(agent_name, (bl, tr), conf, team))

    if DEBUG is not None:
        for image_agent_info in agents:
            bl, tr = image_agent_info.image_location
            agent_name = image_agent_info.name
            conf = image_agent_info.conf
            detection = cv2.rectangle(im, bl, tr, DEBUG)
            cv2.imshow("detection" + str(conf), detection)
            cv2.imshow(agent_name, icons[agent_name][0])
            cv2.waitKey(0)
            cv2.destroyWindow("detection" + str(conf))
            cv2.destroyWindow(agent_name)
    return agents


def get_contours_for_color(color_mse, DEBUG=False):
    ret, threshed = cv2.threshold(color_mse, 90, 255, cv2.THRESH_BINARY_INV)
    threshed = threshed.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if DEBUG:
        thresh_colored = (
            threshed[:, :, None] + np.array([0, 0, 0], dtype=np.uint8)[None, None, :]
        )
        thresh_colored = cv2.drawContours(thresh_colored, contours, -1, (0, 255, 0), 1)
        cv2.imshow(
            "thresh",
            cv2.resize(
                thresh_colored, dsize=(0, 0), fx=8, fy=8, interpolation=cv2.INTER_LINEAR
            ),
        )
        cv2.waitKey(0)
    return contours


def health_contour_score(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    return area - x


def detect_health_bar(
    im,
    image_agent_infos: list[ImageAgentInfo],
    health_bar_y,
    health_bar_width_height,
    health_bar_colors,
    DEBUG=False,
):
    agent_healths = []
    for agent in image_agent_infos:
        # Crop out health bar
        tl = np.copy(agent.image_rect[0])
        tl[1] = health_bar_y
        br = tl + health_bar_width_height
        healthbar_im = im[tl[1] : br[1], tl[0] : br[0]]

        if DEBUG:
            cv2.imshow(
                "healthbar",
                cv2.resize(
                    healthbar_im,
                    dsize=(0, 0),
                    fx=8,
                    fy=8,
                    interpolation=cv2.INTER_LINEAR,
                ),
            )

        mse = np.sum(
            (healthbar_im[None, :, :, :] - health_bar_colors[:, None, None, :]) ** 2,
            axis=-1,
            dtype=np.float32,
        )
        normed_mse = mse
        contours = []
        for color_mse in normed_mse:
            contours += get_contours_for_color(color_mse, DEBUG=DEBUG)

        if len(contours) > 0:
            # get best contour
            cnt = max(contours, key=health_contour_score)
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < 2:
                agent_healths.append(0)
            else:
                health = int(w / 40 * 100)
                agent_healths.append(health)
        else:
            agent_healths.append(0)
    return agent_healths
