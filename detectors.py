import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat
import cv2
import numpy as np
from PIL import Image
from agent import AgentPictureInfo, ScoreboardInfo, ScoreboardPosition
from percep_hash import pHash
from nms import non_max_suppression_fast


def find_all_agents(
    im, icons, agent_image_size, agent_conf_threshold, x_bounds, y_bounds
):
    agents = []
    for agent_name, (hashed, icon, mask) in icons.items():
        confs = cv2.matchTemplate(
            im, icon, cv2.TM_CCOEFF_NORMED, mask=cv2.multiply(mask, 255)
        )
        thresholded_confs = confs > agent_conf_threshold

        # Select bounding boxes by confidences that exceed threshold
        bbox_confs = confs[thresholded_confs]
        (yCoords, xCoords) = (thresholded_confs).nonzero()
        bboxes = []
        for x, y in zip(xCoords, yCoords):
            bboxes.append((x, y, x + agent_image_size[0], y + agent_image_size[1]))
        bboxes = non_max_suppression_fast(bbox_confs, np.array(bboxes), 0.5)

        teams = []
        # Figure out what team each agent is in
        for x1, y1, x2, y2 in bboxes:
            agent_pic = im[y1:y2, x1:x2]
            masked_agent_pic = agent_pic[cv2.multiply(mask, 255) < 254]
            average_color = np.mean(masked_agent_pic, axis=0)
            # (B + G) / 2 > R ?
            if (average_color[0] + average_color[1]) / 2 > average_color[2]:
                team = "blue"
            else:
                team = "red"
            teams.append(team)

        x_cut = x_bounds[0]
        y_cut = y_bounds[0]
        for conf, team, (x1, y1, x2, y2) in zip(bbox_confs, teams, bboxes):
            agents.append(
                AgentPictureInfo(
                    agent_name,
                    np.array([(x1 + x_cut, y1 + y_cut), (x2 + x_cut, y2 + y_cut)]),
                    conf,
                    team,
                )
            )
    return agents


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


def classify_agent(im, icons, agent_conf_threshold):
    imhash = pHash(im)
    for agent_name, (hashed, icon, mask) in icons.items():
        dist = np.count_nonzero(imhash != hashed)
        # print(dist)
        # _, im_portion = pHash(im, debug=True)
        # _, icon_portion = pHash(icon, debug=True)
        # cv2.imshow("icon", icon_portion)
        # cv2.imshow("im", im_portion)
        # cv2.waitKey(0)
        if dist < 40:
            conf = cv2.matchTemplate(
                im, icon, cv2.TM_CCOEFF_NORMED, mask=cv2.multiply(mask, 255)
            ).item()
            if conf > agent_conf_threshold:
                return agent_name, conf
    return None, None


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
        agent_name, conf = classify_agent(im_part, icons, agent_conf_threshold)
        if agent_name is None:
            continue

        agents.append(AgentPictureInfo(agent_name, (bl, tr), conf, team))

    if DEBUG is not None:
        for image_agent_info in agents:
            bl, tr = image_agent_info.image_rect
            agent_name = image_agent_info.agent_name
            conf = image_agent_info.conf
            detection = cv2.rectangle(im, bl, tr, DEBUG)
            cv2.imshow("detection" + str(conf), detection)
            cv2.imshow(agent_name, icons[agent_name][1])
            cv2.waitKey(0)
            cv2.destroyWindow("detection" + str(conf))
            cv2.destroyWindow(agent_name)
    return agents


def get_contours_for_color(color_mse, DEBUG=False):
    ret, threshed = cv2.threshold(color_mse, 50, 255, cv2.THRESH_BINARY_INV)
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
        error = healthbar_im[None, :, :, :].astype(np.float32) - health_bar_colors[:, None, None, :].astype(np.float32)
        mse = np.sum(
            abs(error),
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


def dollar_contour_score(im):
    im_h, im_w = im.shape
    def score(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        s =(-x - abs((im_h/2 - (y+h/2)) ** 2)) + (area<10*-50)+ (h<3*-50)
        # print(x, im_h, s)
        return s
    return score


def read_string(tesseract, im, x_bounds, y_coord, height, filter_dollar=False) -> str:
    cutout = im[y_coord : y_coord + height, x_bounds[0] : x_bounds[1], :]
    cutout = cutout / 255.0
    cutout = cv2.GaussianBlur(cutout, (0, 0), 0.5)

    cutout = cutout[:, :, 0] * cutout[:, :, 1] * cutout[:, :, 2]
    cutout_max = np.max(cutout)
    if cutout_max > 0:
        cutout /= cutout_max
    cutout = (cutout * 255.0).astype(np.uint8)
    if filter_dollar:
        # Filter out the valorant dollar sign so it's not read by tesseract
        edges = cv2.Canny(cutout, 100, 200)
        edges = cv2.dilate(cutout, (3, 3), iterations=2)

        _, edges = cv2.threshold(edges, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
        cntrs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.drawContours(
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cntrs, -1, (0, 255, 0), 1
        )

        if len(cntrs) > 0:
            # get best contour
            cnt = max(cntrs, key=dollar_contour_score(cutout))
            x, y, w, h = cv2.boundingRect(cnt)
            cutout = cv2.rectangle(
                cutout, (x - 1, y - 1), (x + w + 1, y + h + 1), np.median(cutout), -1
            )
        # cv2.namedWindow('contours', cv2.WINDOW_GUI_EXPANDED)
        # cv2.namedWindow('edges', cv2.WINDOW_GUI_EXPANDED)
        # cv2.namedWindow('cutout', cv2.WINDOW_NORMAL)
        # cv2.imshow("contours", contour_image)
        # cv2.imshow("edges", edges)
        # cv2.imshow("cutout", cutout)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    cutout = 255 - cutout
    # _, cutout = cv2.threshold(cutout, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

    h, w = cutout.shape
    tesseract.SetImageBytes(cutout.tobytes(), w, h, 1, w)
    parsed_str = tesseract.GetUTF8Text().strip()
    return parsed_str


def numeric_only(string):
    return re.sub("[^0-9]", "", string)


def read_ultimate(tesseract, im, x_bounds, y_coord, height):
    parsed_str = read_string(tesseract, im, x_bounds, y_coord, height)
    if len(parsed_str) != 3:
        return (1, 0)
    points, limit = parsed_str[0], parsed_str[2]
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
    try:
        return int(numeric_only(parsed_str))
    except:
        print("Warning: Credits not readable")
        return None


def check_spike_status(im, coord, spike_icon, symbol_conf_threshold):
    icon, mask = spike_icon
    h, w, _ = icon.shape
    sample = im[coord[1] : coord[1] + h, coord[0] : coord[0] + w, :]

    result = cv2.matchTemplate(
        sample, icon, cv2.TM_CCOEFF_NORMED, mask=cv2.multiply(mask, 255)
    ).item()
    return result > symbol_conf_threshold


def scoreboard_row(args):
    (
        im,
        spike_icon,
        symbol_conf_threshold,
        tesseract_queue,
        positions,
        entry_height,
    ) = args
    y_coord = positions.y_coord
    tesseract = tesseract_queue.get()
    username = read_string(tesseract, im, positions.username_x, y_coord, entry_height)
    ultimate = read_ultimate(tesseract, im, positions.ultimate_x, y_coord, entry_height)

    kills = read_integer(tesseract, im, positions.kills_x, y_coord, entry_height)
    deaths = read_integer(tesseract, im, positions.deaths_x, y_coord, entry_height)
    assists = read_integer(tesseract, im, positions.assists_x, y_coord, entry_height)
    credits = read_credits(tesseract, im, positions.credits_x, y_coord, entry_height)
    tesseract_queue.put(tesseract)
    spike_status = check_spike_status(
        im, positions.spike_pos, spike_icon, symbol_conf_threshold
    )

    return ScoreboardInfo(
        username, ultimate, kills, deaths, assists, credits, spike_status
    )


def detect_scoreboard(
    im,
    tesseract,
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
    scoreboard_positions = []

    for agent_picture in agent_pictures:
        tl, _ = agent_picture.image_rect
        _, y_coord = tl
        scoreboard_positions.append(
            ScoreboardPosition(
                username_x,
                ultimate_x,
                kills_x,
                deaths_x,
                assists_x,
                credits_x,
                tl - [w, 1],
                y_coord,
            )
        )
    args = list(
        zip(
            repeat(im),
            repeat(spike_icon),
            repeat(symbol_conf_threshold),
            repeat(tesseract),
            scoreboard_positions,
            repeat(entry_height),
        )
    )
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        scoreboard_infos = executor.map(scoreboard_row, args)
    return scoreboard_infos
