# Vim-C-Defines

**Vim-C-Defines** is vim plugin that grays out inactive C/C++/Obj-C preprocessor regions with *pure* python code parsing, NO language server required.
It calculates all define values in a C source project based on my previous work [ukyouz/C-define-Parser][C-define-Parser].
It is intended to be used with C/C++/Obj-C.

It was only tested on Windows but should theoretically work on other platforms, too.

# Pre-Define

If you want to add pre-define specified in runtime build commands, you can add `compile_flags.txt` in your project root. Add `-D` flags for you definitions, like this:

```
-DWIN32
-DENV=1
```


# Configurations

Please see document by command `:help vim-c-defines`


### Requisite

- vim or neovim with python3 support


## Related Work
There are some other plugins providing similar functionality, but in different ways.

* [ifdef highlighting][ifdefhighlighting] adds static vim syntax rules for each *manually* defined macro. It does not make use of a compiler and requires the user to manually specify which macros are defined, thus being rather unflexible and often fails to properly detect skipped regions.

* [DyeVim][DyeVim] integrates with (a custom fork of) [YouCompleteMe][ycm] to retrieve extended syntax information for semantic highlighting, including skipped preprocessor regions.
However, it only works with YCM's libclang completer, not the newer and more advanced clangd completer.

* [vim-lsp-inactive-regions][lspregions] integrates with [vim-lsp][vimlsp] and uses the [cquery][cquery] or [ccls][ccls] language server to retrieve skipped preprocessor regions.

* [vim-lsp-cxx-highlight][vimlspcxx] integrates with various LSP plugins and uses the [cquery][cquery] or [ccls][ccls] language server to provide full semantic highlighting, including skipped preprocessor regions.

* [coc.nvim][coc] + [clangd][coc-clangd] provide semantic highlighting similar to the option above.
  Semantic highlighting support in [coc.nvim][coc] needs to be enabled first, see `:h coc-semantic-highlights`.

* [grayout.vim][grayout.vim] + clang


# References

1. https://github.com/mdda/vim-plugin-python
2. https://devhints.io/vimscript
3. https://pynvim.readthedocs.io/_/downloads/en/stable/pdf/



[C-define-Parser]: https://github.com/ukyouz/C-define-Parser
[ifdefhighlighting]: http://www.vim.org/scripts/script.php?script_id=7
[DyeVim]: https://github.com/davits/DyeVim
[ycm]: https://github.com/ycm-core/YouCompleteMe
[lspregions]: https://github.com/krzbe/vim-lsp-inactive-regions
[vimlsp]: https://github.com/prabirshrestha/vim-lsp
[vimlspcxx]: https://github.com/jackguo380/vim-lsp-cxx-highlight
[compdb]: https://github.com/Sarcasm/compdb
[clangdatabase]: http://clang.llvm.org/docs/JSONCompilationDatabase.html
[bear]: https://github.com/rizsotto/Bear
[cquery]: https://github.com/cquery-project/cquery
[ccls]: https://github.com/MaskRay/ccls
[coc]: https://github.com/neoclide/coc.nvim
[coc-clangd]: https://github.com/clangd/coc-clangd
[grayout.vim]: https://github.com/mphe/grayout.vim
