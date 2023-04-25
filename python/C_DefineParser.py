import logging
import os
import re
import subprocess
# import functools
from collections import OrderedDict, defaultdict, namedtuple
from pprint import pformat

Define = namedtuple(
    "Define",
    ("name", "params", "token", "line", "file", "lineno"),
)
Token = namedtuple("Token", ("name", "params", "line", "span"))

REGEX_TOKEN = re.compile(r"\b(?P<NAME>[a-zA-Z_][a-zA-Z0-9_]+)\b")
REGEX_DEFINE = re.compile(
    r"#define\s+"
    + REGEX_TOKEN.pattern
    + r"(?P<HAS_PAREN>\((?P<PARAMS>[\w, ]*)\))*\s*(?P<TOKEN>.+)*"
)
REGEX_UNDEF = re.compile(r"#undef\s+" + REGEX_TOKEN.pattern)
REGEX_INCLUDE = re.compile(r'#include\s+["<](?P<PATH>.+)[">]\s*')
REGEX_STRING = re.compile(r'"[^"]+"')
REGEX_OPERATOR_NOT = re.compile("!(?!=)")
BIT = lambda n: 1 << n

logger = logging.getLogger("Define Parser")


def glob_recursive(directory, ext=".c"):
    logger.debug("glob **/*%s --recursieve", ext)
    return [
        os.path.join(root, filename)
        for root, dirnames, filenames in os.walk(directory)
        for filename in filenames
        if filename.endswith(ext)
    ]


def is_git(folder):
    markers = {".git", ".gitlab"}
    files = set(os.listdir(folder))
    return len(markers & files)


def git_lsfiles(directory, ext=".h"):
    git_cmds = ["git", "--git-dir", os.path.join(directory, ".git"), "ls-files"]
    logger.debug(" ".join(git_cmds))
    try:
        filelist_output = subprocess.check_output(
            git_cmds,
            shell=True,  # remove flashing empty cmd window prompt
        )
    except subprocess.CalledProcessError:
        # fallback to normal glob if git command fail
        return glob_recursive(directory, ext)
    except FileNotFoundError:
        # fallback to normal glob if git command fail
        return glob_recursive(directory, ext)

    filelist = filelist_output.decode().split("\n")
    return [
        os.path.join(directory, filename)
        for filename in filelist
        if filename.endswith(ext)
    ]


REG_LITERALS = [
    re.compile(r"\b(?P<NUM>[0-9]+)(?:##)?([ul]|ull?|ll?u|ll)\b", re.IGNORECASE),
    re.compile(r"\b(?P<NUM>0b[01]+)(?:##)?([ul]|ull?|ll?u|ll)\b", re.IGNORECASE),
    re.compile(r"\b(?P<NUM>0[0-7]+)(?:##)?([ul]|ull?|ll?u|ll)\b", re.IGNORECASE),
    re.compile(r"\b(?P<NUM>0x[0-9a-f]+)(?:##)?([ul]|ull?|ll?u|ll)\b", re.IGNORECASE),
]

REG_SPECIAL_SIZEOFTYPES = [
    re.compile(r"sizeof\(\s*U8\s*\)"),
    re.compile(r"sizeof\(\s*U16\s*\)"),
    re.compile(r"sizeof\(\s*U32\s*\)"),
    re.compile(r"sizeof\(\s*U64\s*\)"),
]

REG_SPECIAL_TYPES = [
    re.compile(r"\(\s*U8\s*\)"),
    re.compile(r"\(\s*U16\s*\)"),
    re.compile(r"\(\s*U32\s*\)"),
    re.compile(r"\(\s*U64\s*\)"),
]

REG_STATEMENT_IF = re.compile(r"#\s*if(?P<DEF>(?P<NOT>n*)def)*\s*(?P<TOKEN>.+)")
REG_STATEMENT_ELIF = re.compile(r"#\s*elif\s*(?P<TOKEN>.+)")
REG_STATEMENT_ELSE = re.compile(r"#\s*else.*")
REG_STATEMENT_ENDIF = re.compile(r"#\s*endif.*")

REGEX_SYNTAX_LINE_COMMENT = re.compile(r"\s*\/\/.*$")
REGEX_SYNTAX_INLINE_COMMENT = re.compile(r"\/\*[^\/]+\*\/")
REGEX_SYNTAX_LINE_BREAK = re.compile(r"\\\s*$")


class DuplicatedIncludeError(Exception):
    """assert when parser can not found ONE valid include header file."""


class Parser:
    def __init__(self):
        self.reset()
        self.filelines = defaultdict(list)

    def reset(self):
        self.defs = OrderedDict()  # dict of Define
        self.zero_defs = set()
        self.temp_defs = defaultdict(set)  # temp definitions in filename
        self.folder = ""

    def insert_define(self, name, *, params=None, token=None, filename="", lineno=0):
        """params: list of parameters required, token: define body"""
        new_params = params or []
        new_token = token or ""
        self.defs[name] = Define(
            name=name,
            params=new_params,
            token=new_token,
            line="",
            file=filename,
            lineno=lineno,
        )

    def insert_temp_define(
        self, name, *, params=None, token=None, filename="", lineno=0
    ):
        logger.debug("insert temp define: %s", name)
        self.temp_defs[filename].add(name)
        self.insert_define(
            name, params=params, token=token, filename=filename, lineno=lineno
        )

    def remove_temp_define(self, filename):
        logger.debug("remove %d temp defines", len(self.temp_defs[filename]))
        for name in self.temp_defs[filename]:
            if name in self.defs:
                del self.defs[name]
        self.temp_defs[filename] = set()

    def remove_define(self, name):
        if name in self.defs:
            del self.defs[name]
        elif name in self.zero_defs:
            self.zero_defs.remove(name)
        else:
            raise KeyError("token '{}' is not defined!".format(name))

    def strip_token(self, token, reserve_whitespace=False):
        if token == None:
            return None
        if reserve_whitespace:
            token = token.rstrip()
        else:
            token = token.strip()
        token = REGEX_SYNTAX_INLINE_COMMENT.sub("", token)
        return token

    def try_eval_num(self, token):
        # remove integer literals type hint
        for re_reg in REG_LITERALS:
            token = re_reg.sub(r"\1", token)
        # calculate size of special type
        # transform type cascading to bit mask for equivalence calculation
        for data_sz, reg_sizeof_type, reg_special_type in zip(
            [1, 2, 4, 8],
            REG_SPECIAL_SIZEOFTYPES,
            REG_SPECIAL_TYPES
        ):
            # limitation:
            #   for equation like (U32)1 << (U32)(15) may be calculated to wrong value
            #   due to operator order
            # sizeof(U16) -> 2
            token = reg_sizeof_type.sub(str(data_sz), token)
            # (U16)x -> 0xFFFF & x
            token = reg_special_type.sub("0x%s & " % ("F" * data_sz * 2), token)
        # syntax translation from C -> Python
        token = token.replace("/", "//")
        token = token.replace("&&", " and ")
        token = token.replace("||", " or ")
        token = REGEX_OPERATOR_NOT.sub(" not ", token)
        try:
            return int(eval(token))
        except:
            return None

    def read_file_lines(
        self,
        fileio,
        try_if_else=True,
        ignore_header_guard=False,
        reserve_whitespace=False,
        include_block_comment=False,
    ):
        if_depth = 0
        if_true_bmp = 1  # bitmap for every #if statement
        if_done_bmp = 1  # bitmap for every #if statement
        first_guard_token = True
        is_block_comment = False
        # with open(filepath, "r", errors="replace") as fs:
        multi_lines = ""
        for line_no, line in enumerate(fileio.readlines(), 1):

            line = REGEX_SYNTAX_LINE_COMMENT.sub("", self.strip_token(line, reserve_whitespace))

            if not is_block_comment:
                if "/*" in line:  # start of block comment
                    block_comment_start = line.index("/*")
                    is_block_comment = "*/" not in line
                    block_comment_ending = (
                        line.index("*/") + 2 if not is_block_comment else len(line)
                    )
                    line = line[:block_comment_start] + line[block_comment_ending:]
                    if is_block_comment:
                        multi_lines += line

            if is_block_comment:
                if "*/" in line:  # end of block comment
                    line = line[line.index("*/") + 2 :]
                    is_block_comment = False
                else:
                    if include_block_comment:
                        yield (line, line_no)
                    continue

            if try_if_else:
                match_if = REG_STATEMENT_IF.match(line)
                match_elif = REG_STATEMENT_ELIF.match(line)
                match_else = REG_STATEMENT_ELSE.match(line)
                match_endif = REG_STATEMENT_ENDIF.match(line)
                if match_if:
                    if_depth += 1
                    token = match_if.group("TOKEN")
                    if match_if.group("DEF") is not None:
                        # #ifdef, or #ifndef, only need to check whether the definition exists
                        if_tokens = self.find_tokens(token)
                        if_token = (
                            if_tokens[0].name if len(if_tokens) == 1 else "<unknown>"
                        )
                        if (
                            ignore_header_guard
                            and first_guard_token
                            and (match_if.group("NOT") == "n")
                        ):
                            if_token_val = 0  # header guard always uses #ifndef *
                        else:
                            if if_token in self.defs:
                                if not ignore_header_guard:
                                    if_token_val = 1
                                else:
                                    defined_file = self.defs[if_token].file
                                    defined_line = self.defs[if_token].lineno
                                    if os.path.samefile(defined_file, fileio.name) and line_no < defined_line:
                                        if_token_val = 0
                                    else:
                                        if_token_val = 1
                            else:
                                if_token_val = 0
                    else:
                        if_token = self.expand_token(
                            token,
                            try_if_else,
                            raise_key_error=False,
                            zero_undefined=True,
                        )
                        if_token_val = bool(self.try_eval_num(if_token))
                    if_true_bmp |= BIT(if_depth) * (
                        if_token_val ^ (match_if.group("NOT") == "n")
                    )
                    first_guard_token = (
                        False if match_if.group("NOT") == "n" else first_guard_token
                    )
                elif match_elif:
                    if_token = self.expand_token(
                        match_elif.group("TOKEN"),
                        try_if_else,
                        raise_key_error=False,
                        zero_undefined=True,
                    )
                    if_token_val = bool(self.try_eval_num(if_token))
                    if_true_bmp |= BIT(if_depth) * if_token_val
                    if_true_bmp &= ~(BIT(if_depth) & if_done_bmp)
                elif match_else:
                    if_true_bmp ^= BIT(if_depth)  # toggle state
                    if_true_bmp &= ~(BIT(if_depth) & if_done_bmp)
                elif match_endif:
                    if_true_bmp &= ~BIT(if_depth)
                    if_done_bmp &= ~BIT(if_depth)
                    assert if_depth > 0
                    if_depth -= 1

            multi_lines += REGEX_SYNTAX_LINE_BREAK.sub("", line)
            if REGEX_SYNTAX_LINE_BREAK.search(line):
                if reserve_whitespace:
                    if if_true_bmp == BIT(if_depth + 1) - 1:
                        yield (line, line_no)
                continue
            single_line = REGEX_SYNTAX_LINE_BREAK.sub("", multi_lines)
            if if_true_bmp == BIT(if_depth + 1) - 1:
                yield (single_line, line_no)
                if_done_bmp |= BIT(if_depth)
            elif try_if_else and (match_if or match_elif or match_else or match_endif):
                yield (single_line, line_no)
            multi_lines = ""

    def _get_define(self, line, filepath="", lineno=0):
        match = REGEX_UNDEF.match(line)
        if match is not None:
            name = match.group("NAME")
            if name in self.defs:
                del self.defs[name]
            return

        match = REGEX_DEFINE.match(line)
        if match == None:
            return

        name = match.group("NAME")
        parentheses = match.group("HAS_PAREN")
        params = match.group("PARAMS")
        param_list = [p.strip() for p in params.split(",")] if params else []
        match_token = match.group("TOKEN")
        token = self.strip_token(match_token) or "(1)"

        """
        #define AAA     // params = None
        #define BBB()   // params = []
        #define CCC(a)  // params = ['a']
        """
        self.filelines[filepath].append(lineno)
        return Define(
            name=name,
            params=param_list if parentheses else None,
            token=token,
            line=line,
            file=filepath,
            lineno=lineno,
        )

    def read_folder_h(self, directory, try_if_else=True):
        self.folder = directory

        if is_git(directory):
            header_files = git_lsfiles(directory, ".h")
        else:
            header_files = glob_recursive(directory, ".h")
        logger.debug("read_header cnt: %d", len(header_files))

        header_done = set()
        pre_defined_keys = self.defs.keys()

        def get_included_file(inc_path, src_file):
            inc_path = os.path.normpath(inc_path)  # xxx/conf.h
            src_file = os.path.normpath(src_file)  # C:/path/to/src.xxx.c
            included_files = [
                h
                for h in header_files
                if inc_path in h and os.path.basename(inc_path) == os.path.basename(h)
            ]
            if len(included_files) > 1:
                included_files = [f for f in included_files if f.replace(inc_path, "") in src_file]

            if len(included_files) > 1:
                raise DuplicatedIncludeError(
                    pformat(included_files, indent=4, width=120)
                )

            return included_files[0] if len(included_files) else None

        def read_header(filepath):
            if filepath == None or filepath in header_done:
                return

            try:
                with open(filepath, "r", errors="replace") as fs:
                    for line, lineno in self.read_file_lines(fs, try_if_else):
                        match_include = REGEX_INCLUDE.match(line)
                        if match_include is not None:
                            # parse included file first
                            path = match_include.group("PATH")
                            included_file = get_included_file(path, src_file=filepath)
                            read_header(included_file)
                        define = self._get_define(line, filepath, lineno)
                        if define is None or define.name in pre_defined_keys:
                            continue
                        self.defs[define.name] = define

            except UnicodeDecodeError as e:
                logger.warning("Fail to open {!r}. {}".format(filepath, e))

            if filepath in header_files:
                header_done.add(filepath)

        for header_file in header_files:
            read_header(header_file)

        return True

    def find_tokens(self, token):
        def fine_token_params(params):
            if len(params) and params[0] != "(":
                return None
            # (() ())
            brackets = 0
            new_params = ""
            for c in params:
                brackets += (c == "(") * 1 + (c == ")") * -1
                new_params += c
                if brackets == 0:
                    break
            return new_params

        # remove string value in token
        token = REGEX_STRING.sub("", token)

        tokens = list(REGEX_TOKEN.finditer(token))
        if len(tokens):
            ret_tokens = []
            for match in tokens:
                _token = match.group("NAME")
                params = None
                if _token in self.defs and self.defs[_token].params is not None:
                    end_pos = match.end()
                    params = fine_token_params(token[end_pos:])
                param_str = params if params else ""
                ret_tokens.append(
                    Token(name=_token, params=params, line=_token + param_str, span=match.span())
                )
            return ret_tokens
        else:
            return []

    def _check_parentheses(self, token):
        lparan_cnt = 0
        rparan_cnt = 0
        for char in token:
            if char == "(":
                lparan_cnt += 1
            if char == ")":
                rparan_cnt += 1
        return lparan_cnt == rparan_cnt

    def _iter_arg(self, params):
        if len(params) == 0:
            return []
        assert params[0] == "(" and params[-1] == ")"
        parma_list = params[1:-1].split(",")
        arguments = []
        for arg in parma_list:
            arguments.append(arg.strip())
            prams_str = ",".join(arguments)
            if self._check_parentheses(prams_str):
                yield prams_str
                arguments = []

    # @functools.lru_cache
    def expand_token(
        self, token, try_if_else=True, raise_key_error=True, zero_undefined=False
    ):
        expanded_token = self.strip_token(token)

        word_boundary = lambda word: r"\b(##)*%s\b" % re.escape(word)
        tokens = self.find_tokens(expanded_token)
        for _token in tokens:
            name = _token.name
            params = self.strip_token(_token.params)
            if params is not None:
                # Expand all the parameters first
                for p_tok in self.find_tokens(params):
                    params = re.sub(
                        word_boundary(p_tok.line),
                        self.expand_token(
                            p_tok.line, try_if_else, raise_key_error, zero_undefined
                        ),
                        params,
                    )
                    processed = list(t for t in tokens if p_tok.name == t.name)
                    if len(processed):
                        tokens.remove(processed[0])
                if name in self.defs:
                    old_params = self.defs[name].params or []
                    new_params = list(self._iter_arg(params))
                    new_token = self.defs[name].token
                    # Expand the token
                    old_param_regs = (re.compile(word_boundary(x)) for x in old_params)
                    for old_p_reg, new_p in zip(old_param_regs, new_params):
                        new_token = old_p_reg.sub(new_p, new_token)
                    # expanded_token = expanded_token.replace(_token.line, new_token)
                    new_token_val = self.try_eval_num(new_token)
                    new_token = str(new_token_val) if new_token_val else new_token
                    if _token.line == name:
                        expanded_token = re.sub(
                            word_boundary(_token.line), new_token, expanded_token
                        )
                    else:
                        expanded_token = expanded_token.replace(_token.line, new_token)
                    # Take care the remaining tokens
                    expanded_token = self.expand_token(
                        expanded_token, try_if_else, raise_key_error, zero_undefined
                    )
                elif name in self.zero_defs:
                    expanded_token = expanded_token.replace(_token.line, "(0)")
                elif raise_key_error:
                    raise KeyError("token '{}' is not defined!".format(name))
            elif name is not expanded_token:
                params = self.expand_token(
                    _token.line, try_if_else, raise_key_error, zero_undefined
                )
                expanded_token = re.sub(
                    word_boundary(_token.line), params, expanded_token
                )
                # expanded_token = expanded_token.replace(match.group(0), self.expand_token(match.group(0)))

        if expanded_token in self.defs:
            expanded_token = self.expand_token(
                self.defs[token].token, try_if_else, raise_key_error, zero_undefined
            )

            # try to eval the value, to reduce the bracket count
            token_val = self.try_eval_num(expanded_token)
            if token_val is not None:
                expanded_token = str(token_val)
        elif expanded_token in self.zero_defs:
            return "0"
        elif zero_undefined and len(tokens) and expanded_token == name:
            self.zero_defs.add(name)
            return "0"

        return expanded_token

    def get_expand_defines(self, filepath, try_if_else=True, ignore_header_guard=True):
        defines = []

        with open(filepath, "r", errors="replace") as fs:
            for line, lineno in self.read_file_lines(
                fs, try_if_else, ignore_header_guard
            ):
                define = self._get_define(line, filepath, lineno)
                if define == None:
                    continue
                token = self.expand_token(
                    define.token, try_if_else, raise_key_error=False
                )
                if define.name in self.defs:
                    token_val = self.try_eval_num(token)
                    if token_val is not None:
                        self.defs[define.name] = self.defs[define.name]._replace(
                            token=str(token_val)
                        )
                elif define.name in self.zero_defs:
                    token_val = "0"
                defines.append(
                    Define(
                        name=define.name,
                        params=define.params,
                        token=token,
                        line=line,
                        file=filepath,
                        lineno=lineno,
                    )
                )
        return defines

    def get_expand_define(self, macro_name, try_if_else=True):
        if macro_name not in self.defs:
            return None

        define = self.defs[macro_name]
        token = define.token
        expanded_token = self.expand_token(token, try_if_else, raise_key_error=False)

        return Define(
            name=macro_name,
            params=define.params,
            token=expanded_token,
            line=define.line,
            file=define.file,
            lineno=define.lineno,
        )

    def get_preprocess_source(self, filepath, try_if_else=True):
        lines = []

        ignore_header_guard = os.path.splitext(filepath)[1] == ".h"
        with open(filepath, "r", errors="replace") as fs:
            for line, _ in self.read_file_lines(
                fs,
                try_if_else,
                ignore_header_guard,
                reserve_whitespace=True,
            ):
                lines.append(line)
        return lines
