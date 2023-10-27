import io
import logging
import re
import os
import pickle

import C_DefineParser

try:
    import vim
except:
    print("No vim module available outside vim")
    pass


formatter = logging.Formatter(fmt="[{name}] {levelname}: {message}", style="{")

handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger("Define Parser")
logger.setLevel(logging.INFO)


PARSERS = {}
PARSER_IS_BUILDING = set()

HL_SRCID = {}


class Setting:
    Cdf_CacheDirectory = "~/.vim/.Cdf_Cache"
    Cdf_EnableGrayout = True
    Cdf_SupportExtensions = [".h", ".c", ".cpp"]
    Cdf_RootMarkers = [".root", ".git", ".gitlab"]

    Cdf_InactiveRegionHighlightGroup = "comment"


def _escape_filepath(folder):
    trans = str.maketrans("/\\:", "---")
    return folder.translate(trans)


def _convertall_dec2fmt(text, fmt="0x{:X}"):
    re_sub_dec2hex = lambda m: "{}".format(fmt).format(int(m.group(1)))
    return re.sub(r"\b([0-9]+)\b", re_sub_dec2hex, text)


def _get_cache_file_for_folder(folder):
    tag_file = _escape_filepath(folder) + ".dtag"
    cache_folder = vim.command_output("echo expand('%s')" % Setting.Cdf_CacheDirectory)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cache_file = os.path.join(cache_folder, tag_file)
    print("dtag file: %r" % cache_file)
    return cache_file


def _is_root(folder):
    if folder is None:
        return False
    markers = set(Setting.Cdf_RootMarkers)
    files = set(os.listdir(folder))
    return len(markers & files)


def _get_folder():
    return vim.command_output("pwd")


def _get_hl_srcid(bufid):
    if bufid not in HL_SRCID:
        src_id = vim.new_highlight_source()
        HL_SRCID[bufid] = src_id

    return HL_SRCID[bufid]


def _get_configs_from_compile_flags(folder: str):
    compile_flag_txt = os.path.join(folder, "compile_flags.txt")
    if not os.path.exists(compile_flag_txt):
        return []

    with open(compile_flag_txt) as fs:
        arg_txt = fs.read().replace("\n", " ")
    arguments = [x.strip() for x in arg_txt.split(" ") if x]

    predefines = []
    for arg in arguments:
        if not arg.startswith("-D"):
            continue

        pair = tuple(arg[2:].split("="))
        if len(pair) == 1:
            # ie: -DDEBUG
            predefines.append((pair[0], 0))
        elif len(pair) == 2:
            # ie: -DDEBUG=0
            predefines.append(pair)

    return predefines


def _init_parser():
    active_folder = _get_folder()
    if not _is_root(active_folder):
        print("root markers not found in %r, skip!" % active_folder)
        return

    print("init_parser %r" % active_folder)

    cache_file = _get_cache_file_for_folder(active_folder)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as fs:
                PARSERS[active_folder] = pickle.load(fs)
            print("dtag cached found: %r" % cache_file)
            if Setting.Cdf_EnableGrayout:
                _mark_inactive_code(vim.current.buffer)
            return
        except:
            pass

    if active_folder in PARSER_IS_BUILDING:
        return

    PARSER_IS_BUILDING.add(active_folder)

    p = C_DefineParser.Parser()
    PARSERS[active_folder] = p

    # TODO: available to switch configuration
    predefines = _get_configs_from_compile_flags(active_folder)
    for d in predefines:
        # logger.debug("  predefine: %s", d)
        print("  predefine: %r" % d)
        p.insert_define(d[0], token=d[1])

    def async_proc():
        p.read_folder_h(active_folder)
        PARSER_IS_BUILDING.remove(active_folder)

        if Setting.Cdf_EnableGrayout:
            _mark_inactive_code(vim.current.buffer)

        print("done_parser: %r" % active_folder)
        with open(_get_cache_file_for_folder(active_folder), "wb") as fs:
            pickle.dump(p, fs)

    print("building define database, please wait...")
    vim.async_call(async_proc)


def _get_parser():
    active_folder = _get_folder()
    if active_folder not in PARSERS:
        return None
    return PARSERS[active_folder]


def _mark_inactive_code(buffer):
    if _get_folder() in PARSER_IS_BUILDING:
        return
    p = _get_parser()
    if p is None or not buffer.valid:
        return

    filename = buffer.name
    _, ext = os.path.splitext(filename)
    if ext not in Setting.Cdf_SupportExtensions:
        return

    inactive_lines = set(range(1, 1 + len(buffer)))

    fileio = io.StringIO("\n".join(buffer))
    fileio.name = filename
    for _, lineno in p.read_file_lines(
        fileio,
        reserve_whitespace=True,
        ignore_header_guard=True,
        include_block_comment=True,
    ):
        inactive_lines.remove(lineno)
    inactive_lines -= set(p.filelines.get(filename, []))
    print("inactive lines count: %d" % len(inactive_lines))

    regions = [
        (Setting.Cdf_InactiveRegionHighlightGroup, line - 1, 0, -1)
        for line in inactive_lines
    ]
    buffer.update_highlights(_get_hl_srcid(buffer.number), regions)


def _unmark_inactive_code(buffer):
    buffer.clear_highlight(_get_hl_srcid(buffer.number))


def _calc_token(buffer, symbol):
    if _get_folder() in PARSER_IS_BUILDING:
        return
    parser = _get_parser()
    if parser is None or not buffer.valid:
        return

    define = parser.get_expand_define(symbol)
    if define is not None:
        # logger.debug("%r", define)
        value = parser.try_eval_num(define.token)
        if value is not None:
            text = "{} ({})".format(value, hex(value))
        else:
            text = _convertall_dec2fmt(define.token)

        vim.command("echon '\r\r'")
        vim.command("echom '%s = %s'" % (define.name, text))
    else:
        expanded_token = parser.expand_token(symbol)
        # logger.debug("%r", expanded_token)
        value = parser.try_eval_num(expanded_token)
        if value is not None:
            text = "{} ({})".format(value, hex(value))
        else:
            text = _convertall_dec2fmt(expanded_token, "0x{:02x}")
        vim.command("echon '\r\r'")
        for line in symbol.split('\n'):
            vim.command("echom '%s'" % line.lstrip())
        vim.command("echom ' = '")
        for line in text.split('\n'):
            vim.command("echom '%s'" % line.lstrip())


"""
Public Functions
"""


def command_rebuild_define_data(clean_rebuild=False):
    if clean_rebuild:
        cache_file = _get_cache_file_for_folder(_get_folder())
        if os.path.exists(cache_file):
            os.remove(cache_file)
    _init_parser()


def command_mark_inactive_code():
    _mark_inactive_code(vim.current.buffer)


def command_unmark_inactive_code():
    _unmark_inactive_code(vim.current.buffer)


def command_calculate_token(token):
    _calc_token(vim.current.buffer, token)
