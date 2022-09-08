import cv2
import numpy as np
from agent import AgentPictureInfo
from tesserocr import PyTessBaseAPI, PSM, OEM
from PIL import Image
import re


def detect_all_agents(
    im,
    blue_icons,
    red_icons,
    first_blue_agent_pos,
    first_red_agent_pos,
    blue_spacing,
    red_spacing,
    agent_image_size,
    agent_conf_threshold,
    DEBUG=False,
):

    agents = detect_agents_on_team(
        im,
        blue_icons,
        first_blue_agent_pos,
        blue_spacing,
        agent_image_size,
        agent_conf_threshold,
        "blue",
        DEBUG=(255, 0, 0) if DEBUG else None,
    )

    agents += detect_agents_on_team(
        im,
        red_icons,
        first_red_agent_pos,
        red_spacing,
        agent_image_size,
        agent_conf_threshold,
        "red",
        DEBUG=(0, 0, 255) if DEBUG else None,
    )
    return agents


def classify_agent(im, icons):
    agents = dict()
    for agent_name, (icon, mask) in icons.items():
        result = cv2.matchTemplate(im, icon, cv2.TM_CCOEFF_NORMED, mask=cv2.multiply(mask, 255))
        agents[agent_name] = result.flatten()
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
        #     cv2.waitKey(0)
        agent_confs = classify_agent(im_part, icons)

        agent_name, conf = max(agent_confs.items(), key=lambda x: x[1])
        if conf < agent_conf_threshold:
            continue
        agents.append(AgentPictureInfo(agent_name, (bl, tr), conf[0], team))

    if DEBUG is not None:
        for image_agent_info in agents:
            bl, tr = image_agent_info.image_rect
            agent_name = image_agent_info.agent_name
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
    image_agent_infos: list[AgentPictureInfo],
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


def dollar_contour_score(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    return area - x**2


def read_string(tesseract, im, x_bounds, y_coord, height, filter_dollar=False) -> str:
    cutout = im[y_coord : y_coord + height, x_bounds[0] : x_bounds[1], :]
    cutout = cutout / 255.0
    cutout = cutout[:, :, 0] * cutout[:, :, 1] * cutout[:, :, 2]
    cutout /= np.max(cutout)
    cutout = (cutout * 255.0).astype(np.uint8)
    cutout = cv2.GaussianBlur(cutout, (0,0), 0.5)
    if filter_dollar:
        # Filter out the valorant dollar sign so it's not read by tesseract
        edges = cv2.Canny(cutout, 100, 200)
        edges = cv2.dilate(cutout, (3, 3), iterations=2)

        edges = cv2.adaptiveThreshold(
            edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10
        )
        cntrs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.drawContours(
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cntrs, -1, (0, 255, 0), 1
        )
        # cv2.imshow("contours", contour_image)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)
        # cv2.destroyWindow("contours")
        # cv2.destroyWindow("edges")
        if len(cntrs) > 0:
            # get best contour
            cnt = max(cntrs, key=dollar_contour_score)
            x, y, w, h = cv2.boundingRect(cnt)
            cutout = cv2.rectangle(cutout, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cutout = 255 - cutout
    # cv2.imshow("cutout", cutout)
    # cv2.waitKey(0)
    tesseract.SetImage(Image.fromarray(cutout))
    parsed_str = tesseract.GetUTF8Text().strip()
    return parsed_str


def numeric_only(string):
    return re.sub("[^0-9]", "", string)


def read_ultimate(tesseract, im, x_bounds, y_coord, height):
    parsed_str = numeric_only(read_string(tesseract, im, x_bounds, y_coord, height))
    if len(parsed_str) == 0:
        return (1, 0)
    points, limit = parsed_str[0], parsed_str[1]
    try:
        return (int(points), int(limit))
    except:
        print("Warning: ultimate status not readable.")
        return (None, None)


def read_integer(tesseract, im, x_bounds, y_coord, height):
    parsed_str = read_string(tesseract, im, x_bounds, y_coord, height)
    try:
        return int(numeric_only(parsed_str))
    except:
        print("Warning: integer value not readable")
        return None


def read_credits(tesseract, im, x_bounds, y_coord, height):
    parsed_str = read_string(
        tesseract, im, x_bounds, y_coord, height, filter_dollar=True
    )
    return int(numeric_only(parsed_str))


def check_spike_status(im, coord, spike_icon, symbol_conf_threshold):
    icon, mask = spike_icon
    h, w, _ = icon.shape
    sample = im[coord[1] : coord[1] + h, coord[0] : coord[0] + w, :]

    result = cv2.matchTemplate(sample, icon, cv2.TM_CCOEFF_NORMED, mask=cv2.multiply(mask, 255)).item()
    return result > symbol_conf_threshold


def detect_scoreboard(
    im,
    agent_pictures,
    spike_icon,
    symbol_conf_threshold,
    entry_height,
    username_x,
    ultimate_x,
    kills_x,
    deaths_x,
    assists_x,
    credits_x,
):
    h, w, _ = spike_icon[0].shape
    usernames, ultimates, kills, deaths, assists, credits, spike_status = (
        [] for _ in range(7)
    )
    with PyTessBaseAPI(
        path="./tessdata_fast", psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY
    ) as tesseract:
        for agent_picture in agent_pictures:
            tl, _ = agent_picture.image_rect
            _, y_coord = tl
            usernames.append(
                read_string(tesseract, im, username_x, y_coord, entry_height)
            )
            ultimates.append(
                read_ultimate(tesseract, im, ultimate_x, y_coord, entry_height)
            )
            kills.append(read_integer(tesseract, im, kills_x, y_coord, entry_height))
            deaths.append(read_integer(tesseract, im, deaths_x, y_coord, entry_height))
            assists.append(
                read_integer(tesseract, im, assists_x, y_coord, entry_height)
            )
            credits.append(
                read_credits(tesseract, im, credits_x, y_coord, entry_height)
            )
            spike_status.append(
                check_spike_status(im, tl - [w, 1], spike_icon, symbol_conf_threshold)
            )

    return usernames, ultimates, kills, deaths, assists, credits, spike_status
