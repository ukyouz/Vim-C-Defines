if !has("python3")
  echo "vim has to be compiled with +python3 to run this"
  finish
endif

if exists('g:sample_python_plugin_loaded')
  finish
endif

let s:plugin_root_dir = fnamemodify(resolve(expand('<sfile>:p')), ':h')

python3 << EOF
import sys
from os.path import normpath, join
import vim
plugin_root_dir = vim.eval('s:plugin_root_dir')
python_root_dir = normpath(join(plugin_root_dir, '..', 'python'))
sys.path.insert(0, python_root_dir)
import plugin
EOF

"  let g:sample_python_plugin_loaded = 1

function! CdfRebuild()
  python3 plugin.command_rebuild_define_data(v:true)
endfunction

command! -bang -nargs=0 CdfRebuild call CdfRebuild()

function! CdfRefreshBuffer()
  python3 plugin.command_mark_inactive_code()
endfunction

command! -bang -nargs=0 CdfRefreshBuffer call CdfRefreshBuffer()

"  function! CdfTest()
"    python3 plugin.command_mark_inactive_code()
"  endfunction

"  function! CdfTest2()
"    python3 plugin.command_unmark_inactive_code()
"  endfunction

augroup cdf_autoload
  autocmd!
  autocmd DirChanged * silent python3 plugin.command_rebuild_define_data()
  autocmd BufReadPost * silent python3 plugin.command_mark_inactive_code()
  autocmd BufWritePost * silent python3 plugin.command_mark_inactive_code()
augroup END