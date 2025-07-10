from pyswip import Prolog
from nle.nethack import actions
from colorama import Fore, Style, init
import re

init(autoreset=True)

nethack_to_colorama = {
    0: Style.RESET_ALL,      # Black/background/default
    1: Fore.RED,
    2: Fore.GREEN,
    3: Fore.YELLOW,
    4: Fore.BLUE,
    5: Fore.MAGENTA,
    6: Fore.CYAN,
    7: Fore.WHITE,
    8: Style.BRIGHT + Fore.BLACK,   # Bold/gray
    9: Style.BRIGHT + Fore.RED,
    10: Style.BRIGHT + Fore.GREEN,
    11: Style.BRIGHT + Fore.YELLOW,
    12: Style.BRIGHT + Fore.BLUE,
    13: Style.BRIGHT + Fore.MAGENTA,
    14: Style.BRIGHT + Fore.CYAN,
    15: Style.BRIGHT + Fore.WHITE,
}


def save(pred, name, prolog):
    stringa=""
    results = list(prolog.query(pred))
    for item in results:
        stringa+=name+"("
        for key in item:
            tmp = item[key]
            if type(tmp)==int:
                tmp = str(tmp)
            pos = (tmp).find(",")
            if pos ==0:
                stringa+=tmp.replace(",","",1)+","
            else:
                stringa+=tmp+","
        stringa=stringa[:len(stringa)-1]
        stringa+=")\n"
    return stringa

def load(file, prolog):
    with open(file, "r") as f:
        str = f.read()
    list = str.split("\n")
    for s in list:
        if s.strip():
            prolog.assertz(s)

def a_star(start, goal, agent):
    from heapq import heappush, heappop

    open_set = []  # priority queue: (f(n), position)
    heappush(open_set, (0, start))

    came_from = {}  # maps each node to its predecessor
    cost_so_far = {start: 0}  # g(n): cost from start to node

    print(f"[A*] Start: {start}, Goal: {goal}")

    while open_set:
        _, current = heappop(open_set)  # node with lowest f(n)

        if current == goal:
            # reconstruct path from goal to start
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nx, ny in agent.neighbors(current):  # get walkable neighbors
            neighbor = (nx, ny)
            new_cost = cost_so_far[current] + 1  # cost to reach this neighbor

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + agent.heuristic(current, neighbor, goal)  # f(n) = g(n) + h(n)
                heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    return None

def ascii_to_idx(char):
    """
    Given a character (e.g. 'e'), returns the index of the corresponding action in actions.ACTIONS.
    """
    val = ord(char)
    for idx, action in enumerate(actions.ACTIONS):
        if hasattr(action, 'value') and action.value == val:
            return idx
    return None

def print_inventory(obs):
    inv_letters = obs["inv_letters"]
    inv_strs = obs["inv_strs"]
    array = []
    for idx, c in enumerate(inv_letters):
        if c != 0:
            letter = chr(c)

            obj_name = "".join([chr(x) for x in inv_strs[idx] if x != 0]).strip()
            array.append(f"  {letter}: {obj_name}")

    return array




def Simboli_unici(observation):
    
    tty_chars=observation['tty_chars']
    tty_colors=observation['tty_colors']

    height = len(tty_chars)
    tty_chars = tty_chars[1:height-2]
    tty_colors = tty_colors[1:height-2]
    symbols = set()


    for row_chars, row_colors in zip(tty_chars, tty_colors):
        for char, color in zip(row_chars, row_colors):
                symbols.add((char, color))

    return symbols


def quote(s):
    if s is None:
        return 'none'
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("'", "''")
    if re.search(r"[^\w]", s):
        return f"'{s}'"
    return s

def inventoryToProlog(lst, prolog):
    for line in lst:
        match = re.search(r"([a-z]): (an?|[\d]+)? ?(blessed|uncursed|cursed)? ?(\+\d+)? ?(.*?)( \((.*?)\))?$", line)
        if match:
            id, article, state, bonus, obj, _, note = match.groups()
            try:
                quantity = int(article)
            except:
                quantity = 1

           
            state = quote(state)
            bonus = quote(bonus)
            obj = quote(obj)
            flag = "false" if note is None else "true"
            obj = "healing" if obj.find("healing")!=-1 else obj
            obj = "potion" if obj.find("potion")!=-1 else obj
            prolog.assertz(f"has({obj}, {state}, {quantity}, {bonus}, {flag}, {id})")
